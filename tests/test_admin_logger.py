"""
Admin event logging: the audit trail, and keeping it off the request path.

The bug this file exists to prevent: the logger read `SERVER_TIMESTAMP` off a
Firestore `Client`, so every write raised `AttributeError` and the whole
admin_events trail was silently empty. Fixing the sentinel turned the no-op
into real blocking network I/O, which then stalled the SSE response after its
last event. Both halves have to hold at once: the record must be written, and
writing it must never hold a qualification open.
"""

import threading
import unittest
from unittest import mock

import admin_logger


class _FakeCollection:
    def __init__(self, sink):
        self._sink = sink

    def add(self, document):
        self._sink.append(document)


class _FakeFirestore:
    def __init__(self, sink, sentinel=object(), block=None):
        self._sink = sink
        self.SERVER_TIMESTAMP = sentinel
        self._block = block

    def collection(self, name):
        return _FakeCollection(self._sink)


class ServerTimestampTest(unittest.TestCase):
    def test_the_sentinel_is_not_read_off_the_client_instance(self):
        """The original defect: `db.SERVER_TIMESTAMP` on a Client instance. The
        client here explodes on that attribute, and the module carries the real
        sentinel, so a correct implementation still writes."""
        from google.cloud import firestore

        sink = []

        class ClientWithoutSentinel:
            @property
            def SERVER_TIMESTAMP(self):
                raise AttributeError(
                    "'Client' object has no attribute 'SERVER_TIMESTAMP'")

            def collection(self, name):
                return _FakeCollection(sink)
        with mock.patch.object(admin_logger, "_get_firestore",
                               return_value=ClientWithoutSentinel()):
            self.assertTrue(admin_logger.log_admin_event_sync("e", "u", {}))

        self.assertEqual(sink[0]["timestamp"], firestore.SERVER_TIMESTAMP)

    def test_a_write_records_the_event(self):
        sink = []
        with mock.patch.object(admin_logger, "_get_firestore",
                               return_value=_FakeFirestore(sink, "SENTINEL")):
            with mock.patch.object(admin_logger, "_server_timestamp",
                                   return_value="SENTINEL"):
                self.assertTrue(admin_logger.log_admin_event_sync(
                    "ask_message_received", "user-1", {"a": 1}))

        self.assertEqual(len(sink), 1)
        self.assertEqual(sink[0]["type"], "ask_message_received")
        self.assertEqual(sink[0]["userId"], "user-1")
        self.assertEqual(sink[0]["metadata"], {"a": 1})
        self.assertEqual(sink[0]["timestamp"], "SENTINEL")

    def test_a_write_that_fails_does_not_raise(self):
        class Broken:
            def collection(self, name):
                raise RuntimeError("firestore down")

        with mock.patch.object(admin_logger, "_get_firestore", return_value=Broken()):
            self.assertFalse(admin_logger.log_admin_event_sync("e", "u", {}))

    def test_a_missing_client_is_not_an_error(self):
        with mock.patch.object(admin_logger, "_get_firestore", return_value=None):
            self.assertFalse(admin_logger.log_admin_event_sync("e", "u", {}))


class NeverBlocksTheCallerTest(unittest.TestCase):
    """`_log_usage` runs after every model call, including from inside the SSE
    async generator. A slow Firestore must not be able to hold a request open."""

    def test_log_token_usage_returns_before_the_write_completes(self):
        release = threading.Event()
        entered = threading.Event()
        sink = []

        class SlowFake(_FakeFirestore):
            def collection(self, name):
                entered.set()
                release.wait(5)
                return _FakeCollection(sink)

        with mock.patch.object(admin_logger, "_get_firestore",
                               return_value=SlowFake(sink)):
            result = admin_logger.log_token_usage("e", "u", {"total_tokens": 9})

        self.assertTrue(result, "the caller must not wait on Firestore")
        self.assertTrue(entered.wait(5), "the write should still be attempted")
        self.assertEqual(sink, [], "the write has not completed yet")
        release.set()

    def test_the_write_still_happens(self):
        sink = []
        done = threading.Event()

        class SignallingFake(_FakeFirestore):
            def collection(self, name):
                inner = _FakeCollection(self._sink)

                class Wrapper:
                    def add(self, document):
                        inner.add(document)
                        done.set()

                return Wrapper()

        with mock.patch.object(admin_logger, "_get_firestore",
                               return_value=SignallingFake(sink)):
            admin_logger.log_token_usage("e", "u", {"total_tokens": 9})

        self.assertTrue(done.wait(5), "the audit record was never written")
        self.assertEqual(sink[0]["metadata"]["total_tokens"], 9)

    def test_the_async_wrapper_does_not_block(self):
        release = threading.Event()
        entered = threading.Event()

        class SlowFake(_FakeFirestore):
            def collection(self, name):
                entered.set()
                release.wait(5)
                return _FakeCollection([])

        import asyncio
        with mock.patch.object(admin_logger, "_get_firestore",
                               return_value=SlowFake([])):
            self.assertTrue(asyncio.run(
                admin_logger.log_admin_event("e", "u", {})))
        self.assertTrue(entered.wait(5))
        release.set()

    def test_the_worker_thread_is_a_daemon(self):
        """A non-daemon thread would keep the process alive at shutdown."""
        with mock.patch.object(admin_logger, "_write_admin_event"):
            admin_logger.log_token_usage("e", "u", {"total_tokens": 1})
        for thread in threading.enumerate():
            if thread.name == "admin-logger":
                self.assertTrue(thread.daemon)
                return
        self.skipTest("thread finished before it could be inspected")

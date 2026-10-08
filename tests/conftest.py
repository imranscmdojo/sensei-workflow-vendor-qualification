"""
Test bootstrap. Runs before any test module is imported.

The one job here is to stop the suite from touching a real database.
`main.py` calls `load_dotenv()` at import, so without this the tests would pick
up `SUPPLIER_PORTAL_DB=./data/portal.db` from `.env` and `reset_all()` would
delete the developer's live supplier sessions. That is not hypothetical — it
happened, and it wiped 199 rows.

An empty value is enough: `load_dotenv` does not override variables that already
exist, even when they exist but are empty. Tests therefore run memory-only,
which is what `reset_all()` and the restart tests assume.
"""

import os

os.environ["SUPPLIER_PORTAL_DB"] = ""
os.environ.pop("SUPPLIER_PORTAL_BASE_URL", None)
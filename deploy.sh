#!/bin/bash
# Deploy the Vendor Qualification backend to Cloud Run.
#
#   ./deploy.sh
#   gcloud run services describe vendor-qualification-backend --region us-west1
#
# Modeled on sensei-workflow-supplier-contract-backend/deploy.sh. The only
# secret passed is the SMTP password, read from .env (gitignored) at run
# time: this service authenticates with the attached service account
# (Vertex AI + Firestore audit writes) and verifies Firebase ID tokens itself.
set -e
cd "$(dirname "$0")"

PROJECT_ID="test-rag-corpus-project"
FIREBASE_PROJECT_ID="sensei-ask-project"
REGION="us-west1"
SERVICE_NAME="vendor-qualification-backend"
# Shared corpus. Same one Supplier Contract Review uses - nothing is
# re-ingested for this tool.
CORPUS="projects/test-rag-corpus-project/locations/us-west1/ragCorpora/137359788634800128"

# Local overrides win, so a developer can deploy against their own frontend or
# pin a different model. Every variable below is still set explicitly below,
# which is what actually reaches Cloud Run.
if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "Deploying $SERVICE_NAME to Cloud Run..."

# `--set-env-vars` splits its value list on commas, so ALLOWED_ORIGINS — itself
# a comma-separated origin list — blew it up with "Bad syntax for dict arg".
# `--env-vars-file` takes YAML, where a comma is just text in a quoted string.
ENV_FILE="$(mktemp)"
trap 'rm -f "$ENV_FILE"' EXIT
{
  echo "PROJECT_ID: \"$PROJECT_ID\""
  echo "FIREBASE_PROJECT_ID: \"$FIREBASE_PROJECT_ID\""
  echo "LOCATION: \"$REGION\""
  echo "RAG_CORPUS: \"$CORPUS\""
  echo "RAG_RETRIEVAL_MODEL: \"${RAG_RETRIEVAL_MODEL:-gemini-2.5-flash}\""
  echo "RAG_GENERATION_MODEL: \"${RAG_GENERATION_MODEL:-gemini-2.5-pro}\""
  echo "ALLOWED_ORIGINS: \"${ALLOWED_ORIGINS:-https://www.scmsensei.ai}\""
  echo "AUTH_DISABLED: \"false\""
  # Invite/report link base + durable portal store + invitation SMTP.
  # Hardcoded on purpose: .env carries localhost values that must never
  # reach production. SMTP credentials come from .env (never the repo).
  echo "SUPPLIER_PORTAL_BASE_URL: \"https://nh.scmsensei.ai\""
  echo "SUPPLIER_PORTAL_DB: \"/tmp/supplier_portal.db\""
  echo "SMTP_HOST: \"${SMTP_HOST:-}\""
  echo "SMTP_PORT: \"${SMTP_PORT:-587}\""
  echo "SMTP_USER: \"${SMTP_USER:-}\""
  echo "SMTP_PASS: \"${SMTP_PASS:-}\""
  echo "SMTP_FROM: \"${SMTP_FROM:-}\""
  echo "SMTP_USE_TLS: \"${SMTP_USE_TLS:-1}\""
  echo "NOTIFICATION_TO: \"${NOTIFICATION_TO:-}\""
} > "$ENV_FILE"

gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --concurrency 40 \
  --timeout 3600 \
  --env-vars-file "$ENV_FILE"

SERVICE_URL="https://$SERVICE_NAME-214011868588.$REGION.run.app"

echo "Deployment complete!"
echo "Service URL: $SERVICE_URL"
echo "Health check:"
echo "  curl $SERVICE_URL/health"
echo
echo "Set this in front-nextjs/.env.local when pointing the app at the live backend:"
echo "  NEXT_PUBLIC_VENDOR_QUALIFICATION_BACKEND=$SERVICE_URL"

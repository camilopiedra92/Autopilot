#!/bin/bash
gcloud run deploy bank-to-ynab \
  --image=us-central1-docker.pkg.dev/antigravity-bank-ynab/mcp-cloud-run-deployments/bank-to-ynab \
  --region=us-central1 \
  --project=antigravity-bank-ynab \
  --platform=managed \
  --allow-unauthenticated \
  --memory=512Mi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=5 \
  --concurrency=80 \
  --timeout=120s \
  --set-env-vars=PYTHONUNBUFFERED=1,GOOGLE_CLOUD_PROJECT=antigravity-bank-ynab,GCP_PUBSUB_TOPIC=projects/antigravity-bank-ynab/topics/gmail-notifications,GMAIL_SENDER_FILTER=alertasynotificaciones \
  --set-secrets=GOOGLE_API_KEY=google-api-key:latest,YNAB_ACCESS_TOKEN=ynab-access-token:latest,API_KEY_SECRET=api-key-secret:latest,TODOIST_API_TOKEN=todoist-api-token:latest,AIRTABLE_PERSONAL_ACCESS_TOKEN=airtable-personal-access-token:latest,TELEGRAM_BOT_TOKEN=telegram-bot-token:latest,/secrets/credentials/credentials.json=gmail-credentials:latest,/secrets/token/token.json=gmail-token:latest \
  --cpu-boost

# Deploy to Cloud Run

Goal: a public link that answers questions from the FAA handbook chapters.

## Step 0: build the saved index (on your computer, PowerShell, key set)
    python scripts/build_index.py
This makes data/phak/index.npz. Commit it (it is about 5 MB). The server loads it so it only embeds the question.

## Step 1: one-time Google Cloud setup
1. Create a project at console.cloud.google.com (name: rag-assistant).
2. Link billing. Then set a budget alert at 10 dollars (Billing > Budgets).
3. Install the gcloud CLI, then: gcloud auth login
4. gcloud config set project YOUR_PROJECT_ID
5. gcloud services enable run.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

## Step 2: store the Gemini key as a secret (never in the image or git)
    echo -n "YOUR_KEY" | gcloud secrets create gemini-key --data-file=-
(Use the key only in your own terminal. Never paste it into chat.)

Let Cloud Run read it:
    gcloud projects add-iam-policy-binding YOUR_PROJECT_ID --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"

## Step 3: deploy
    gcloud run deploy rag-assistant --source . --region us-central1 --allow-unauthenticated --set-secrets GEMINI_API_KEY=gemini-key:latest --set-env-vars ROUTER_MODEL=gemini-3.5-flash-lite,GENERATION_MODEL=gemini-3.5-flash-lite --max-instances 2 --memory 1Gi

## Step 4: test
    curl https://YOUR-URL/healthz
    curl -X POST https://YOUR-URL/ask -H "Content-Type: application/json" -d "{\"question\":\"What does the altimeter show if the static port is blocked?\"}"

## Cost and safety
- max-instances 2 and a 6 requests per minute per IP limit protect the free Gemini quota.
- Cloud Run scales to zero, so idle cost is 0.
- To take it down: gcloud run services delete rag-assistant --region us-central1

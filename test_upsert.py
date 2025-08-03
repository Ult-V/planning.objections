import requests
import json
import google.generativeai as genai
from google.cloud import secretmanager

# --- Configuration ---
GCP_PROJECT_ID = "crucial-accord-467816-g0"
INDEX_ID = "8241710463390318592" # Paste your actual Index ID here
GCP_LOCATION = "europe-west2"

def get_secret(project_id, secret_id, version_id="latest"):
    """Fetches a secret from Google Cloud Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def upsert_via_rest(embeddings, access_token):
    """Saves the embeddings to Vector Search using a direct REST API call."""
    print("\n--- Testing Upsert to Vector Search ---")

    # CORRECTED URL: Targets the 'indexes' path
    endpoint_url = f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/indexes/{INDEX_ID}:upsertDatapoints"

    datapoints = [{"datapointId": "test_id_1", "featureVector": embedding} for embedding in embeddings]
    request_body = {"datapoints": datapoints}

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(endpoint_url, headers=headers, data=json.dumps(request_body))

    if response.status_code == 200:
        print("✅ Successfully upserted datapoints.")
    else:
        print(f"❌ Error upserting to Vector Search:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Fetch secrets
    print("🔐 Fetching secrets...")
    GOOGLE_API_KEY = get_secret(GCP_PROJECT_ID, "google-api-key")
    ACCESS_TOKEN = get_secret(GCP_PROJECT_ID, "gcp-access-token")
    print("✅ Secrets fetched.")

    # 2. Generate a single test embedding
    print("\n--- Generating test embedding ---")
    genai.configure(api_key=GOOGLE_API_KEY)
    test_embedding = genai.embed_content(
        model="models/text-embedding-004",
        content="This is a test",
        task_type="RETRIEVAL_DOCUMENT"
    )['embedding']
    print("✅ Test embedding generated.")

    # 3. Call the upsert function with the test embedding
    upsert_via_rest([test_embedding], ACCESS_TOKEN)
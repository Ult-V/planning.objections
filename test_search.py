import json
import google.generativeai as genai
import requests
from google.cloud import secretmanager

# --- Config ---
GCP_PROJECT_ID = "crucial-accord-467816-g0"
INDEX_ENDPOINT_ID = "1298285737891856384"
DEPLOYED_INDEX_ID = "islington_policy_endpoint_1754245523158"
GCP_LOCATION = "europe-west2"

def get_secret(project_id, secret_id, version_id="latest"):
    """Fetch a secret from Google Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def search_vector_index(query_text, access_token, api_key):
    """Embed a query and search the Vertex AI Vector Search index."""
    try:
        genai.configure(api_key=api_key)
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
    except Exception as e:
        return f"❌ Error generating embedding: {e}"

    endpoint_url = (
        f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID}:findNeighbors"
    )

    request_body = {
        "deployedIndexId": DEPLOYED_INDEX_ID,
        "queries": [
            {
                "datapoint": {"featureVector": query_embedding},
                "neighborCount": 1
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(endpoint_url, headers=headers, data=json.dumps(request_body))

    if response.status_code == 200:
        return response.json()
    else:
        print("❌ Error during index search:")
        print("Status Code:", response.status_code)
        print("Response:", response.text)
        return None

# --- Entry point ---
if __name__ == "__main__":
    print("🔐 Fetching secrets...")
    GOOGLE_API_KEY = get_secret(GCP_PROJECT_ID, "google-api-key")
    ACCESS_TOKEN = get_secret(GCP_PROJECT_ID, "gcp-access-token")
    print("✅ Secrets fetched.")

    test_query = "The design respects the surrounding local character and scale."
    print(f"\n🔍 Testing query: '{test_query}'")

    result = search_vector_index(test_query, ACCESS_TOKEN, GOOGLE_API_KEY)

    print("\n--- Search Result ---")
    if isinstance(result, dict):
        print(json.dumps(result, indent=2))
    else:
        print(result)

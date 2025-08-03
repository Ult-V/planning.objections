import os
import json
import google.generativeai as genai
import requests
from google.cloud import secretmanager

# --- Configuration ---
GCP_PROJECT_ID = "crucial-accord-467816-g0"
INDEX_ENDPOINT_ID = "islington_policy_endpoint_1754245523158"
GCP_LOCATION = "europe-west2"

def get_secret(project_id, secret_id, version_id="latest"):
    """Fetches a secret from Google Cloud Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def search_vector_index(query_text, access_token, api_key):
    """Takes a query, embeds it, and searches the Vector Search index."""
    print(f"\n--- Searching for policies related to: '{query_text}' ---")

    # 1. Generate embedding for the user's query
    try:
        genai.configure(api_key=api_key)
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text,
            task_type="RETRIEVAL_QUERY" # Use RETRIEVAL_QUERY for search
        )['embedding']
        print("✅ Query embedding generated.")
    except Exception as e:
        print(f"❌ Error generating query embedding: {e}")
        return None

    # 2. Prepare the direct API call
    endpoint_url = f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID}:findNeighbors"

    # ADD THE DEPLOYED_INDEX_ID HERE
    request_body = {
        "deployedIndexId": "islington_policy_endpoint_1754245523158",
        "queries": [{
            "datapoint": {"featureVector": query_embedding},
            "neighborCount": 3 # Find the top 3 matches
        }]
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # 3. Make the API call
    response = requests.post(endpoint_url, headers=headers, data=json.dumps(request_body))

    # 4. Process and print the results
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Error searching index:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return None

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load the text chunks from the file
    try:
        with open("policy_chunks.json", "r") as f:
            policy_chunks = json.load(f)
    except FileNotFoundError:
        print("❌ Error: policy_chunks.json not found. Please run extract_text.py first.")
        exit()

    # Fetch secrets
    print("🔐 Fetching secrets...")
    GOOGLE_API_KEY = get_secret(GCP_PROJECT_ID, "google-api-key")
    ACCESS_TOKEN = get_secret(GCP_PROJECT_ID, "gcp-access-token")
    print("✅ Secrets fetched.")

    # Define a sample user objection to search for
    SAMPLE_QUERY = "The proposed building is too tall and will block sunlight."

    # Search the index
    search_results = search_vector_index(SAMPLE_QUERY, ACCESS_TOKEN, GOOGLE_API_KEY)

    # Display the results
    if search_results:
        print("\n✅ Found Top 3 Matching Policies:")
        for neighbor in search_results['nearestNeighbors'][0]['neighbors']:
            # Get the ID and use it to look up the text in our loaded file
            match_id = int(neighbor['datapoint']['datapointId'])
            match_text = policy_chunks[match_id]
            match_distance = neighbor['distance']

            print("\n----------------------------------")
            print(f"MATCH (Distance: {match_distance:.4f}):")
            print(match_text)
            print("----------------------------------")
import os
import json
import fitz
import google.generativeai as genai
from google.cloud import aiplatform
from google.cloud import secretmanager

# --- Configuration ---
GCP_PROJECT_ID = "crucial-accord-467816-g0"
INDEX_ENDPOINT_ID = "1298285737891856384"
DEPLOYED_INDEX_ID = "islington_policy_endpoint_1754245523158"
GCP_LOCATION = "europe-west2"
DAS_PDF_FILENAME = "513FE6CDE1C811EC824B005056865ECD.pdf"

def get_secret(project_id, secret_id, version_id="latest"):
    """Fetches a secret from Google Cloud Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def extract_text_from_pdf(full_pdf_path):
    """Opens a PDF and extracts all text content."""
    try:
        doc = fitz.open(full_pdf_path)
        print(f"✅ Successfully opened '{os.path.basename(full_pdf_path)}'.")
        full_text = "".join(page.get_text("text") for page in doc)
        doc.close()
        return full_text
    except Exception as e:
        print(f"❌ Error processing PDF file: {e}")
        return None

def chunk_text(text, chunk_size=2000, overlap=400):
    """Splits a long text into larger chunks for summarization."""
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def summarize_claims(text_chunks, api_key):
    """Uses Gemini to summarize key claims from DAS text chunks."""
    print("\n--- Summarizing developer claims from the DAS ---")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        summarized_claims = []
        for i, chunk in enumerate(text_chunks[:3]): # Process first 3 chunks for PoC
            print(f"Processing chunk {i+1}...")
            prompt = f"Read the following text from a Design and Access Statement. Summarize the developer's main claim or promise in a single, concise sentence. Text: '{chunk}'"
            response = model.generate_content(prompt)
            summarized_claims.append(response.text.strip())
        print("✅ Claims summarized successfully.")
        return summarized_claims
    except Exception as e:
        print(f"❌ Error summarizing claims: {e}")
        return None

def search_vector_index(query_text, api_key):
    """Uses the Vertex AI SDK to embed and search the vector index."""
    try:
        genai.configure(api_key=api_key)
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
    except Exception as e:
        return f"Error generating embedding: {e}"

    try:
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=INDEX_ENDPOINT_ID)

        result = index_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_embedding],
            num_neighbors=1, # Corrected parameter name
        )
        # The result is a list containing one list of neighbors
        return result[0]
    except Exception as e:
        return f"Error searching index via SDK: {e}"

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        with open("policy_chunks.json", "r") as f:
            policy_chunks = json.load(f)
    except FileNotFoundError:
        print("❌ Error: policy_chunks.json not found. Please run extract_text.py first.")
        exit()

    print("🔐 Fetching secrets...")
    GOOGLE_API_KEY = get_secret(GCP_PROJECT_ID, "google-api-key")
    # The SDK uses Application Default Credentials, so the access token is no longer needed here
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    print("✅ Secrets fetched and AI Platform initialized.")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        das_pdf_path = os.path.join(script_dir, '..', 'data sources for Islington', DAS_PDF_FILENAME)
    except NameError:
        das_pdf_path = os.path.join('..', 'data sources for Islington', DAS_PDF_FILENAME)

    das_text = extract_text_from_pdf(das_pdf_path)
    if das_text:
        das_chunks = chunk_text(das_text)
        claims = summarize_claims(das_chunks, GOOGLE_API_KEY)

        if claims:
            print("\n--- Potential Conflicts Found ---")
            for claim in claims:
                print(f"\n[Developer Claim]: {claim}")
                search_results = search_vector_index(claim, GOOGLE_API_KEY)

                if isinstance(search_results, list) and len(search_results) > 0:
                    neighbor = search_results[0]
                    match_id = int(neighbor.id)
                    match_text = policy_chunks[match_id]
                    print(f"[Relevant Policy Found (Distance: {neighbor.distance:.4f})]: {match_text.strip()}")
                else:
                    print(f"[No relevant policy found or error in search]: {search_results}")
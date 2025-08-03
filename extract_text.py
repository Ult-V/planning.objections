import fitz
import os
import google.generativeai as genai
from google.cloud import aiplatform
import requests
import json

# --- Configuration ---
GOOGLE_API_KEY = "AIzaSyDm5H5W6p5FkzvzgXG5fyxkx6ni6cNUYF8"
ACCESS_TOKEN = "PASTE_YOUR_TOKEN_HERE" # The token you just copied
GCP_PROJECT_ID = "crucial-accord-467816-g0"
GCP_LOCATION = "europe-west2"
INDEX_ID = "6651939794928533504" # The ID of the INDEX, not the endpoint
INDEX_ENDPOINT_ID = "6320925222316802048" # Get this from the GCP Console
PDF_FILENAME = "islington-council-local-plan-strategic-and-development-management-policies.pdf"

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, '..', 'data sources for Islington', PDF_FILENAME)
except NameError:
    pdf_path = os.path.join('..', 'data sources for Islington', PDF_FILENAME)


def extract_text_from_pdf(full_pdf_path):
    try:
        doc = fitz.open(full_pdf_path)
        print(f"✅ Successfully opened '{os.path.basename(full_pdf_path)}'. Pages: {doc.page_count}")
        full_text = "".join(page.get_text("text") for page in doc)
        doc.close()
        print("✅ Text extraction complete.")
        return full_text
    except Exception as e:
        print(f"❌ Error processing PDF file: {e}")
        return None

def chunk_text(text, chunk_size=1000, overlap=200):
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    print(f"✅ Text split into {len(chunks)} chunks.")
    return chunks

def generate_embeddings(chunks_to_embed):
    print("\n--- Generating Embeddings ---")
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=chunks_to_embed,
            task_type="RETRIEVAL_DOCUMENT"
        )
        print(f"✅ Successfully generated {len(result['embedding'])} embeddings.")
        return result['embedding']
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return None

def upsert_via_rest(embeddings):
    """Saves the embeddings to Vector Search using a direct REST API call."""
    print("\n--- Upserting to Vector Search (Direct API Call) ---")
    
    # Construct the regional API endpoint URL
    endpoint_url = f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID}:upsertDatapoints"
    
    # Format the data for the REST API
    datapoints = [{"datapointId": str(i), "featureVector": embedding} for i, embedding in enumerate(embeddings)]
    request_body = {"datapoints": datapoints}
    
    # Set the authorization header
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Make the POST request
    response = requests.post(endpoint_url, headers=headers, data=json.dumps(request_body))
    
    # Check the result
    if response.status_code == 200:
        print("✅ Successfully upserted datapoints.")
    else:
        print(f"❌ Error upserting to Vector Search:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")


# --- Main Execution ---
if __name__ == "__main__":
    if os.path.exists(pdf_path):
        MAX_CHUNKS_TO_PROCESS = 50
        policy_text = extract_text_from_pdf(pdf_path)
        
        if policy_text:
            text_chunks = chunk_text(policy_text)[:MAX_CHUNKS_TO_PROCESS]
            
            if text_chunks:
                embeddings_result = generate_embeddings(text_chunks)
                if embeddings_result:
                    upsert_via_rest(embeddings_result)
    else:
        print(f"❌ Error: The file was not found at the expected path: {pdf_path}")
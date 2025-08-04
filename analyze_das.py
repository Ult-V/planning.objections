import os
import json
import fitz
import google.generativeai as genai
from google.cloud import aiplatform
from google.cloud import secretmanager
from fpdf import FPDF
from fpdf.enums import XPos, YPos # New import for updated syntax

# --- Configuration ---
GCP_PROJECT_ID = "crucial-accord-467816-g0"
INDEX_ENDPOINT_ID = "1298285737891856384"
DEPLOYED_INDEX_ID = "islington_policy_endpoint_1754245523158"
GCP_LOCATION = "europe-west2"
DAS_PDF_FILENAME = "513FE6CDE1C811EC824B005056865ECD.pdf"

def get_secret(project_id, secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def extract_text_from_pdf(full_pdf_path):
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
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def summarize_claims(text_chunks, api_key):
    print("\n--- Summarizing developer claims from the DAS ---")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        summarized_claims = []
        for i, chunk in enumerate(text_chunks[:3]):
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
            num_neighbors=1,
        )
        return result[0]
    except Exception as e:
        return f"Error searching index via SDK: {e}"

def analyze_compliance(developer_claim, policy_text, api_key):
    """Uses Gemini to compare a claim against a policy for non-compliance."""
    print("--- Analyzing for non-compliance ---")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        prompt = f"""
        You are an expert planning assistant, highly critical of developer proposals in London. Your goal is to find valid grounds for objection based on common planning concerns.

        Here is a claim from a developer's Design and Access Statement (DAS) and the most relevant planning policy.

        **Developer's Claim:**
        "{developer_claim}"

        **Relevant Planning Policy:**
        "{policy_text}"

        First, analyze the developer's claim. Is it specific with verifiable details, or is it a vague, aspirational statement?

        Next, compare the claim and the policy against this checklist of common objection themes for London estate regenerations:
        - **Affordability & Housing Mix:** Does the claim fail to specify the number of social rented homes? Is the term 'affordable' used without a clear definition that matches the policy?
        - **Density & Overdevelopment:** Does the policy mention appropriate density or scale? Does the developer's claim sound like it could lead to overcrowding, overshadowing, or loss of light?
        - **Design & Heritage:** Does the policy call for respecting local character? Does the developer's claim use generic terms like 'high-quality design' without providing specific details that prove it respects the area's heritage?
        - **Green Space & Trees:** Does the policy protect green space or trees? Is the developer's claim about 'enhanced green space' vague and potentially misleading?

        Based on this critical analysis, identify the most likely reason for an objection. Summarize this specific point of non-compliance or vagueness in one paragraph.
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error during compliance analysis: {e}"

def save_results_to_pdf(results_data):
    """Takes a list of claims and analyses and saves them to a PDF."""
    print("\n--- Generating PDF Report ---")
    pdf = FPDF()
    # Add the Unicode font
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf") # Assumes you download the bold version too for titles
    pdf.add_page()

    # Use the Unicode font
    pdf.set_font("DejaVu", 'B', 16)
    # Use modern syntax to avoid warnings
    pdf.cell(0, 10, text="Planning Objection Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)

    for result in results_data:
        pdf.set_font("DejaVu", 'B', 12)
        pdf.multi_cell(0, 5, text=f"[Developer Claim]: {result['claim']}")
        pdf.ln(5)
        pdf.set_font("DejaVu", '', 12)
        pdf.multi_cell(0, 5, text=f"[Compliance Analysis]: {result['analysis']}")
        pdf.ln(10)

    pdf_output_path = "objection_report.pdf"
    pdf.output(pdf_output_path)
    print(f"✅ Report saved to {pdf_output_path}")


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
            print("\n--- Potential Objection Points Found ---")
            report_data = []
            for claim in claims:
                print(f"\n[Developer Claim]: {claim}")
                search_results = search_vector_index(claim, GOOGLE_API_KEY)

                if isinstance(search_results, list) and len(search_results) > 0:
                    neighbor = search_results[0]
                    match_id = int(neighbor.id)
                    match_text = policy_chunks[match_id]

                    analysis_result = analyze_compliance(claim, match_text, GOOGLE_API_KEY)
                    print(f"[Compliance Analysis]: {analysis_result}")
                    report_data.append({"claim": claim, "analysis": analysis_result})
                else:
                    print(f"[No relevant policy found or error in search]: {search_results}")

            if report_data:
                # To make the bold font work, you also need to download 'DejaVuSans-Bold.ttf'
                # and place it in the same folder.
                # You can get it here: https://github.com/dejavu-fonts/dejavu-fonts/blob/master/ttf/DejaVuSans-Bold.ttf?raw=true
                save_results_to_pdf(report_data)
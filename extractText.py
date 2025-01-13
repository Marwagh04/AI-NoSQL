import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import pinecone

# Initialize Pinecone (no environment parameter required for serverless)
pinecone.init(api_key="pcsk_3iUfRX_F1cPojuDopWJBtTgPYqjG6Fv9EDJrQLTGtJiZxrmn2UNUetXJixRGgsQRXHrCi6")
index_name = "pdf-document-vectors"

# Create Pinecone index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)  # Dimension depends on your embedding model
index = pinecone.Index(index_name)

# Initialize SentenceTransformer for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths
base_dir = "data"  # Adjusted base directory for your structure
processed_dir = os.path.join(base_dir, "processed")

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

# Function to index text in Pinecone
def index_text_in_pinecone(file_name, text):
    try:
        # Generate embedding for the text
        vector = model.encode(text).tolist()
        # Upload to Pinecone with metadata
        index.upsert([(file_name, vector, {"file_name": file_name})])
        print(f"Indexed: {file_name}")
    except Exception as e:
        print(f"Error indexing {file_name}: {e}")

# Process PDFs in the processed directory
def process_pdfs_and_store_in_pinecone():
    for root, _, files in os.walk(processed_dir):
        for file_name in files:
            if file_name.endswith(".pdf"):
                file_path = os.path.join(root, file_name)
                print(f"Processing: {file_path}")
                text = extract_text_from_pdf(file_path)
                if text:
                    index_text_in_pinecone(file_name, text)

# Run the pipeline
if __name__ == "__main__":
    process_pdfs_and_store_in_pinecone()

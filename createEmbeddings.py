import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import re

# Initialize Pinecone instance
pc = Pinecone(api_key="pcsk_uRWBd_9d9dj1Y23zC6WzU6uC1iZJWqKXhCVUEECGtquP7JmJPTNzzhQd2Cgs7k2HpBNYe")  # Replace with your actual API key

# Index configuration
index_name = "txt-document-vectors"  # Use the name from your Pinecone UI
base_dir = "data"  # Base directory for your text files
texts_dir = os.path.join(base_dir, "processed", "texts")  # Path to the texts folder

# Create Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension based on `all-MiniLM-L6-v2`
        metric="cosine",  # Similarity metric (cosine)
        spec=ServerlessSpec(
            cloud="aws",  # Cloud provider
            region="us-east-1"  # Region from the Pinecone UI
        )
    )

# Connect to the Pinecone index
index = pc.Index(index_name)

# Initialize SentenceTransformer for embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# Function to index text in Pinecone
def index_text_in_pinecone(file_name, text):
    # Create an ASCII-safe ID by removing non-ASCII characters
    safe_file_name = re.sub(r'[^a-zA-Z0-9]', '_', file_name)  # Replace non-ASCII chars with underscores
    # Try to fetch the record and check if it already exists
    existing_record = index.fetch([safe_file_name])
    if existing_record.ids:  # If record exists, skip indexing
        print(f"Record for {safe_file_name} already exists. Skipping...")
    else:
        try:
            # Generate embedding for the text
            vector = model.encode(text).tolist()

            # Upload to Pinecone with metadata
            index.upsert(vectors=[{"id": safe_file_name, "values": vector, "metadata": {"file_name": file_name}}])
            print(f"Indexed: {safe_file_name}")

        except Exception as e:
            print(f"Error indexing {file_name}: {e}")


# Process text files in the 'texts' directory
def process_text_files_and_store_in_pinecone():
    for root, _, files in os.walk(texts_dir):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = os.path.join(root, file_name)
                print(f"Processing: {file_path}")
                try:
                    # Read text from the file
                    with open(file_path, "r", encoding="utf-8") as file:
                        text = file.read()
                    if text:
                        index_text_in_pinecone(file_name, text)
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")

# Run the pipeline
if __name__ == "__main__":
    process_text_files_and_store_in_pinecone()

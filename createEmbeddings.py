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
    # Recreate the Pinecone index with a larger dimension
    pc.create_index(
        name=index_name,
        dimension=384,
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


# Function to split text into smaller chunks
def split_text(text, chunk_size=500):  # Chunk size in characters
    # Split the text into chunks of the specified size
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Process each chunk separately
def index_text_in_pinecone(file_name, text):
    # Split text into chunks if it's too large
    text_chunks = split_text(text)

    for i, chunk in enumerate(text_chunks):
        # Create an ASCII-safe ID by including the chunk index to avoid duplicates
        safe_file_name = re.sub(r'[^a-zA-Z0-9]', '_', f"{file_name}_{i}")
        
        try:
            # Generate embedding for the chunk
            vector = model.encode(chunk).tolist()

            # Upload to Pinecone with metadata for the chunk
            index.upsert(vectors=[{"id": safe_file_name, "values": vector, "metadata": {"file_name": file_name, "chunk_index": i, "text": chunk}}])
            print(f"Indexed: {safe_file_name}")

        except Exception as e:
            print(f"Error indexing chunk {i} of {file_name}: {e}")


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

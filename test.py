import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import openai
import numpy as np

# Initialize Pinecone instance with API key
pinecone_api_key = "pcsk_uRWBd_9d9dj1Y23zC6WzU6uC1iZJWqKXhCVUEECGtquP7JmJPTNzzhQd2Cgs7k2HpBNYe"
pc = Pinecone(api_key=pinecone_api_key)

# Index configuration
index_name = "txt-document-vectors"
index = pc.Index(index_name)

# Initialize SentenceTransformer for embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Set your OpenAI API key directly
openai.api_key = "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"

# Function to generate a summary based on user-provided key points
def generate_summary(key_points, num_key_points=3):
    summaries = []
    
    for i in range(num_key_points):
        prompt = key_points[i]
        
        # Generate and normalize the query vector
        query_vector = model.encode([prompt])
        query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize to unit vector
        query_vector=[str(float(x)) for x in query_vector.flatten()]
        # query_vector = [round(v, 6) for v in query_vector.flatten()]  # Round to 6 decimal places
        # Debugging: Print the query vector to check its format and dimension
        print(f"Query Vector for prompt '{prompt}': {query_vector[:10]}...")  # Print first 10 values for inspection
        print(f"Query Vector Length: {len(query_vector)}")

        if len(query_vector) != 384:
            print(f"Error: Query vector dimension is incorrect (expected 384, got {len(query_vector)})")
            continue

        try:
            # Search for relevant chunks based on the prompt
            results = index.query(queries=[query_vector], top_k=1)  # Retrieve the top 1 result for each prompt
            
            if not results or 'results' not in results or len(results['results']) == 0:
                print(f"No results found for the prompt: {prompt}")
                context = "No relevant text found."
            else:
                print(f"Pinecone results for '{prompt}': {results}")
                context = "\n".join([match['metadata']['text'] for match in results['results'][0]['matches']])
            
            # Use OpenAI GPT-3.5-turbo to generate the summary for the key point
            full_prompt = f"Summarize the following information related to '{prompt}':\n{context}"
            
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=full_prompt,
                max_tokens=150,
                temperature=0.7
            )
            summary = response['choices'][0]['text'].strip()
        except Exception as e:
            summary = f"Error generating summary: {str(e)}"

        summaries.append({"key_point": prompt, "summary": summary})
    
    return summaries

# Prompts aligned with Bayes' Theorem
key_points = [
    "What is Bayes' Theorem and its key concepts?",
    "How is Bayes' Theorem applied in real-world problems?",
    "Explain the mathematical formulation of Bayes' Theorem."
]

# Generate summaries
summaries = generate_summary(key_points, num_key_points=3)
for summary in summaries:
    print(f"Key Point: {summary['key_point']}")
    print(f"Summary: {summary['summary']}\n")

import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
import google.generativeai as genai

# Initialize Pinecone instance with API key
pinecone_api_key = "pcsk_uRWBd_9d9dj1Y23zC6WzU6uC1iZJWqKXhCVUEECGtquP7JmJPTNzzhQd2Cgs7k2HpBNYe"
pc = Pinecone(api_key=pinecone_api_key)

# Index configuration
index_name = "txt-document-vectors"
index = pc.Index(index_name)

# Initialize SentenceTransformer for embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Configure Google Generative AI with the API key
genai_api_key = "AIzaSyD_bmrFMFMTISMu0160udu474ZOdxMoR7A"
genai.configure(api_key=genai_api_key)

# Initialize the GenerativeModel
genai_model = genai.GenerativeModel('gemini-1.5-flash')  # Replace with the desired model name


# Function to generate a answer based on user-provided key points
def generate_answer(key_points, num_key_points=3):
    answers = []
    
    for i in range(num_key_points):
        prompt = key_points[i]
        
        # Generate and normalize the query vector
        query_vector = model.encode([prompt])
        query_vector = query_vector / np.linalg.norm(query_vector)  # Normalize to unit vector

        # Flatten the query_vector (ensure it's 1D)
        query_vector = query_vector.flatten().tolist()

        # Debugging: Print the query vector to check its format and dimension
        #print(f"Query Vector for prompt '{prompt}': {query_vector[:10]}...")  # Print first 10 values for inspection
        #print(f"Query Vector Length: {len(query_vector)}")

        if len(query_vector) != 384:
            print(f"Error: Query vector dimension is incorrect (expected 384, got {len(query_vector)})")
            continue

        try:
            # Search for relevant chunks based on the prompt
            results = index.query(namespace="",vector=query_vector,top_k=40,include_values=True,includeMetadata=True)
            if not results or 'matches' not in results or len(results['matches']) == 0:
                print(f"No results found for the prompt: {prompt}")
                context = "No relevant text found."
            else:
                #print(f"Pinecone results for '{prompt}': {results}")
                # Extract the `text` field from the metadata of the top match
                context = results['matches'][0]['metadata']['text']
            
            # Use Google Generative AI to generate the summary for the key point
            full_prompt = f"Answer to the following question'{prompt}':\n using these documents{context}"
            
            # Use the GenerativeModel instance to generate the content
            response = genai_model.generate_content(full_prompt)
            answer = response.text.strip()  # Extract and clean the response text
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

        answers.append({"key_point": prompt, "answer": answer})
    
    return answers


# Prompts aligned with Bayes' Theorem
key_points = [
    "What is Bayes' Theorem and its key concepts?",
    "Explain Bayes' Theorem",
    "Explain the mathematical formulation of Bayes' Theorem."
]

# Generate summaries
answers = generate_answer(key_points, num_key_points=3)
for answer in answers:
    print(f"Key Point: {answer['key_point']}")
    print(f"Answer: {answer['answer']}\n")

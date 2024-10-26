import os
import requests
from sentence_transformers import SentenceTransformer
import faiss
import cohere
import numpy as np
import json
import nltk
from nltk.tokenize import sent_tokenize

nltk.data.path.append('C:\\Users\\prach\\nltk_data')  # Specify your NLTK data path

# API Keys
translation_api_key = os.getenv("TRANSLATION_API_KEY")
cohere_client = cohere.Client("zmL9dmSpg01YK2qBBTA2lSS9wxC9FxzKEBxsDcvY")
ASR_ENDPOINT = "http://localhost:8000/transcribe/"

# Initialize the model for embedding generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, effective

# Load the FAISS index for vector search
faiss_index = faiss.read_index("vector_database.faiss")

# 1. Audio Transcription by calling ASR Endpoint
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(ASR_ENDPOINT, files=files)
        if response.status_code == 200:
            transcription = response.json().get("transcription")
            return transcription if transcription else "No transcription available"
        else:
            raise Exception(f"Transcription Error: {response.status_code} - {response.text}")

# 2. Translation Function using Sarvam's API
def translate_to_hindi(text, api_key):
    url = "https://api.sarvam.ai/translate"
    body = {
        "input": text,
        "source_language_code": "en-IN",
        "target_language_code": "hi-IN",  # Hindi language code
        "speaker_gender": "Female",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": True
    }
    headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    response = requests.post(url, json=body, headers=headers)
    if response.status_code == 200:
        translated_data = response.json()
        return translated_data.get("translated_text", "Translation not found.")
    else:
        raise Exception(f"Translation Error: {response.status_code}, {response.text}")

# 3. Retrieve Top-2 Closest Chunks from Vector Database
def retrieve_relevant_chunks(query):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = faiss_index.search(query_embedding, 2)  # Retrieve top-2 matches
    return indices.flatten().tolist()



def get_answer(question, context_chunks):
    # Initialize the prompt using the question and the context
    prompt = f"Question: {question}\nContext:\n{context_chunks[0]}\n{context_chunks[1]}\nAnswer:"

    # API Endpoint for Cohere Chat
    url = "https://api.cohere.com/v2/chat"

    headers = {
        "Authorization": "Bearer zmL9dmSpg01YK2qBBTA2lSS9wxC9FxzKEBxsDcvY",  # Use your API key here
        "Content-Type": "application/json"
    }

    # Request payload based on your example
    payload = {
        "model": "command-r",  # Use a model that your account has access to
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    # Sending the request
    response = requests.post(url, headers=headers, json=payload)

    # Handling the response
    if response.status_code == 200:
        response_data = response.json()
        print(f"Response Data: {response_data}")  # Debug: print full response for inspection

        # Check if response_data contains the message key
        if "message" in response_data:
            message_data = response_data["message"]
            if "content" in message_data and isinstance(message_data["content"], list):
                # Extract the text from the first item in the content list
                content_list = message_data["content"]
                if len(content_list) > 0 and "text" in content_list[0]:
                    return content_list[0]["text"].strip()
                else:
                    print("Warning: No valid content found in the response.")
            else:
                print("Warning: No valid content key found in message data.")
        else:
            print("Error: Unexpected response structure - missing 'message' key.")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        raise Exception(f"Error: {response.status_code}, {response.text}")

    return None  # If no valid answer is found


def rag_pipeline(audio_file):
    # Step 1: Transcribe audio to text
    question_text = transcribe_audio(audio_file)
    
    # Ensure question_text is a single string if it's a list
    if isinstance(question_text, list):
        question_text = " ".join(question_text)
    
    print(f"Transcribed Question: {question_text}")
    
    # Step 2: Translate question text to Hindi (assuming you still need this step)
    translated_text = translate_to_hindi(question_text, translation_api_key)
    print(f"Translated Text: {translated_text}")

    # Step 3: Retrieve relevant context from vector database
    relevant_indices = retrieve_relevant_chunks(translated_text)
    
    # Step 4: Load the chunks and filter valid indices
    with open("chunks.json", "r") as f:  # Load chunks previously saved as JSON
        chunks = json.load(f)
    
    # Validate and select only available indices
    context_chunks = [chunks[idx] for idx in relevant_indices if idx < len(chunks)]
    
    # Add a default text if not enough context chunks are found
    if len(context_chunks) < 2:
        print("Warning: Not enough context chunks found. Filling with default text.")
        while len(context_chunks) < 2:
            context_chunks.append("No relevant information found.")
    
    # Step 5: Generate the answer using LLM (updated part)
    answer = get_answer(translated_text, context_chunks)
    
    return answer

# Example usage
if __name__ == "__main__":
    audio_file_path = "question_audio.wav"  # Replace with path to the audio file
    answer = rag_pipeline(audio_file_path)
    print("Final Answer:", answer)








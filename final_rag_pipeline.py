import os
import requests
from sentence_transformers import SentenceTransformer
import faiss
import cohere
import numpy as np
import json
import nltk
from nltk.tokenize import sent_tokenize

nltk.data.path.append('C:\\Users\\prach\\nltk_data')  # Set NLTK data path

#------------------------------------ API Configuration --------------------------------------
translation_api_key = os.getenv("TRANSLATION_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client("cohere_api_key")
ASR_ENDPOINT = "http://localhost:8000/transcribe/"

#------------------------------------ Model and FAISS Initialization --------------------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model for vector representation
faiss_index = faiss.read_index("vector_database.faiss")    # Load pre-built FAISS index for vector search

#------------------------------------ Audio Transcription --------------------------------------
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(ASR_ENDPOINT, files=files)
        if response.status_code == 200:
            transcription = response.json().get("transcription")
            return transcription if transcription else "No transcription available"
        else:
            raise Exception(f"Transcription Error: {response.status_code} - {response.text}")

#------------------------------------ Text Translation (Using Sarvam API) --------------------------------------
def translate_to_hindi(text, api_key):
    url = "https://api.sarvam.ai/translate"
    body = {
        "input": text,
        "source_language_code": "en-IN",
        "target_language_code": "hi-IN",
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

#------------------------------------ Retrieve Relevant Chunks from Vector Database --------------------------------------
def retrieve_relevant_chunks(query):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = faiss_index.search(query_embedding, 2)  # Retrieve top-2 matches
    return indices.flatten().tolist()

import os

#------------------------------------ Generate Answer using Cohere API --------------------------------------
def get_answer(question, context_chunks):
    # Construct prompt with question and context
    prompt = f"Question: {question}\nContext:\n{context_chunks[0]}\n{context_chunks[1]}\nAnswer:"
    url = "https://api.cohere.com/v2/chat"

    # Retrieve Cohere API key from environment variable
    cohere_api_key = os.getenv("COHERE_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {cohere_api_key}",  # Use the environment variable for the API key
        "Content-Type": "application/json"
    }
    payload = {
        "model": "command-r",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    response = requests.post(url, headers=headers, json=payload)

    # Handle and parse response
    if response.status_code == 200:
        response_data = response.json()
        if "message" in response_data:
            message_data = response_data["message"]
            if "content" in message_data and isinstance(message_data["content"], list):
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

    return None  # Return None if no valid answer is found


#------------------------------------ RAG Pipeline --------------------------------------
def rag_pipeline(audio_file):
    # Step 1: Transcribe audio to text
    question_text = transcribe_audio(audio_file)
    
    if isinstance(question_text, list):  # Ensure question_text is a single string
        question_text = " ".join(question_text)
    
    print(f"Transcribed Question: {question_text}")
    
    # Step 2: Translate question text to Hindi
    translated_text = translate_to_hindi(question_text, translation_api_key)
    print(f"Translated Text: {translated_text}")

    # Step 3: Retrieve relevant context from vector database
    relevant_indices = retrieve_relevant_chunks(translated_text)
    
    # Step 4: Load and validate context chunks
    with open("chunks.json", "r") as f:
        chunks = json.load(f)
    
    context_chunks = [chunks[idx] for idx in relevant_indices if idx < len(chunks)]
    
    if len(context_chunks) < 2:  # Fill with default text if not enough context chunks
        print("Warning: Not enough context chunks found. Filling with default text.")
        while len(context_chunks) < 2:
            context_chunks.append("No relevant information found.")
    
    # Step 5: Generate answer using LLM
    answer = get_answer(translated_text, context_chunks)
    
    return answer

#------------------------------------ Execution --------------------------------------
if __name__ == "__main__":
    audio_file_path = "question_audio.wav"  # Path to the audio file
    answer = rag_pipeline(audio_file_path)
    print("Final Answer:", answer)




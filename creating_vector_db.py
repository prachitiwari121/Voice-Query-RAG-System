import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.data.path.append('C:\\Users\\prach\\nltk_data')
from nltk.tokenize import sent_tokenize

# 1. Chunk the Text
def chunk_text(text, chunk_size=300, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(chunk))
            chunk = chunk[-overlap:]  # Overlap the chunks
            current_length = sum(len(s.split()) for s in chunk)

        chunk.append(sentence)
        current_length += sentence_length

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

# 2. Generate Embeddings using Hugging Face model
def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, and effective embedding model
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings

# 3. Store embeddings in FAISS
def store_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    index.add(embeddings)  # Add embeddings to the FAISS index
    return index

# 4. Save FAISS index
def save_faiss_index(index, file_name):
    faiss.write_index(index, file_name)
    print(f"FAISS index saved to {file_name}")

# Main function to orchestrate the process
def main():
    # Simulated scraped Wikipedia article content
    with open("AI.txt", "r", encoding="utf-8") as file:
        scraped_text = file.read()

    # Step 1: Chunk the text
    chunks = chunk_text(scraped_text)
    print(f"Text chunked into {len(chunks)} parts.")

    # Step 2: Generate embeddings for each chunk
    embeddings = generate_embeddings(chunks)
    print(f"Generated embeddings for {len(embeddings)} chunks.")

    # Step 3: Store embeddings in FAISS index
    faiss_index = store_in_faiss(embeddings)

    # Step 4: Save FAISS index to a file
    save_faiss_index(faiss_index, "vector_database.faiss")

if __name__ == "__main__":
    main()

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.data.path.append('C:\\Users\\prach\\nltk_data')
from nltk.tokenize import sent_tokenize

#------------------------------------ Text Chunking --------------------------------------
def chunk_text(text, chunk_size=300, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(chunk))
            chunk = chunk[-overlap:]  # Overlap chunks for continuity
            current_length = sum(len(s.split()) for s in chunk)

        chunk.append(sentence)
        current_length += sentence_length

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

#------------------------------------ Embedding Generation --------------------------------------
def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model for sentence vectors
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings

#------------------------------------ FAISS Index Storage --------------------------------------
def store_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance metric for similarity
    index.add(embeddings)  # Add embeddings to FAISS index
    return index

#------------------------------------ Save FAISS Index to File --------------------------------------
def save_faiss_index(index, file_name):
    faiss.write_index(index, file_name)
    print(f"FAISS index saved to {file_name}")

#------------------------------------ Main Process --------------------------------------
def main():
    # Load text data (scraped content from Wikipedia)
    with open("AI.txt", "r", encoding="utf-8") as file:
        scraped_text = file.read()

    # Step 1: Chunk the text
    chunks = chunk_text(scraped_text)
    print(f"Text chunked into {len(chunks)} parts.")

    # Step 2: Generate embeddings
    embeddings = generate_embeddings(chunks)
    print(f"Generated embeddings for {len(embeddings)} chunks.")

    # Step 3: Store embeddings in FAISS index
    faiss_index = store_in_faiss(embeddings)

    # Step 4: Save FAISS index
    save_faiss_index(faiss_index, "vector_database.faiss")

if __name__ == "__main__":
    main()

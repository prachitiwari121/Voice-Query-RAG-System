# Voice-Enabled Conversational System

This project implements an end-to-end voice-enabled conversational system, incorporating **Automatic Speech Recognition (ASR)**, **text translation**, **vector database retrieval**, and **Retrieval-Augmented Generation (RAG)** to answer questions based on provided audio input.

## Project Structure

- **`asr_app.py`**: FastAPI-based ASR module that transcribes audio files using NeMo's Conformer model.
- **`creating_vector_db.py`**: Builds a vector database from text chunks to enable retrieval for question answering.
- **`data_collection.py`**: Collects relevant Wikipedia data based on search queries and saves it to a text file for vector database creation.
- **`final_rag_pipeline.py`**: Combines ASR, translation, retrieval, and Cohere API for RAG, completing the question-answer process from audio input.

## Setup and Installation

### Prerequisites
- **Python 3.8+**
- **NVIDIA GPU** with CUDA (for ASR model if available)
- Required Python packages listed in `requirements.txt`

### Install Dependencies

```bash
pip install -r requirements.txt
```

## API Keys
Set up environment variables for API keys as follows:

- `GOOGLE_API_KEY` and `SEARCH_ENGINE_ID` for Wikipedia search.
- `TRANSLATION_API_KEY` for Sarvam API.
- `COHERE_API_KEY` for Cohere API.

## Model and NLTK Data
- Download the ASR model (`ai4b_indicConformer_hi.nemo`) and update its path in `asr_app.py`.
- Download NLTK data and set the path in `creating_vector_db.py` and `final_rag_pipeline.py`.

## Running the System

### Run Data Collection
Use `data_collection.py` to gather text data from Wikipedia:

```bash
python data_collection.py --query "<SEARCH QUERY>"
```

### Create Vector Database
Run `creating_vector_db.py` to create a vector database from the collected text:

```bash
python creating_vector_db.py
```

### Start ASR Server
Start the ASR server using `asr_app.py`:

```bash
uvicorn asr_app:app --host 0.0.0.0 --port 8000
```

### Run the RAG Pipeline
Run `final_rag_pipeline.py` with an audio file to complete the RAG pipeline and generate an answer:

```bash
python final_rag_pipeline.py
```

## Project Workflow

- **Audio Input**: Accepts audio files and transcribes them using ASR.
- **Translation**: Translates the transcription (if needed) to the target language using Sarvam API.
- **Vector Database Retrieval**: Finds relevant content using FAISS for vector search.
- **Answer Generation**: Generates responses using Cohere's API with the retrieved context.


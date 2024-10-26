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

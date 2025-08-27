# Mini RAG (Retrieval-Augmented Generation) in Python

## Overview

This project demonstrates a simple RAG pipeline using Python and SpaCy. It retrieves relevant documents from a local dataset based on a question, computes concept-based vector similarities, and generates a context prompt for answering questions.

---

## Features

- Tokenizes documents and questions using SpaCy.
- Filters tokens by part-of-speech (POS).
- Computes concept frequency vectors per document.
- Computes cosine similarity between a question and documents.
- Returns top relevant documents to construct a context prompt.

---

## Requirements

- Python 3.9+
- SpaCy
- spaCy model `en_core_web_md`

Install dependencies with:

```bash
pip install spacy
python -m spacy download en_core_web_md
```

---

## Project Structure

```bash
.
├── datas/                # Folder containing your text documents
├── mini_rag.py           # Main script
└── README.md
```

---

## Usage

1. Place your documents (plain text) in the datas/ folder.
1. Update question in the script or modify to take user input.
1. Run the script:
```bash
python mini_rag.py
```
1. The script prints the prompt including top relevant documents and the question.

---

## Configuration
- `DATA_PATH`: Path to your documents folder.
- `POS_FILTER`: List of token types to include (default: nouns, verbs, adjectives, etc.).
- `THRESHOLD_SIMILARITY`: Minimum similarity to include a document.
- `SPACY_MODEL`: SpaCy model for tokenization and embeddings.
- `TOP_K`: Maximum number of documents to include in the prompt (default: 3).

---

## Notes
- Tokenization is case-insensitive.
- Currently uses raw frequency vectors; consider TF-IDF for better weighting.
- Only the top TOP_K most similar documents are included in the prompt.
- Large documents may need truncation for prompt limits.

---

## License
MIT License



Notes:
Provided source in ground truth is not correct
Recall@5 to be calculated after chunking by Item instead of pages
Calling aple doc for question for apple and tesla doc for tesla question instead of calling from a combined doc
Table of content is creating noise such as "Item 1B unresolved comments"



# Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers complex financial and legal questions using these documents:

- Apple Inc. 2024 Form 10-K
- Tesla, Inc. 2023 Form 10-K

The system retrieves relevant sections from the filings and generates accurate, well-sourced answers using a locally hosted LLM.

---
# Notebook with all modules imported and runnable:
https://colab.research.google.com/drive/1O2nDkuLSe4mY-f1aPtGBRCTtfiKEIUXP?usp=sharing

---

## Features

- PDF ingestion and structured chunking
- Metadata preservation (document, section, page number)
- Open-source embeddings (BAAI/bge-base-en-v1.5)
- FAISS vector database
- Cross-encoder reranking (BAAI/bge-reranker-base)
- Local LLM (Mistral-7B-Instruct-v0.3)

---

# Steps

## Step 1 — Ingestion & Indexing

- The PDFs are loaded using `PyPDFLoader` library
- Recursive character chunking with chunk size 1400 and overlap 100
- Secondary splitting by `Item X` headers
- Metadata added:
  - `document` (Apple 10-K / Tesla 10-K)
  - `item` (Item 1B, etc.)
  - `page` (Page number - index 1)
- Embeddings generated using:
  - `BAAI/bge-base-en-v1.5`
- Indexed in:
  - FAISS vector store

---

## Step 2 — Retrieval

1. Top-20 similarity search
2. Cross-encoder reranking using:
   - `BAAI/bge-reranker-base`
3. Top-5 reranked chunks passed to LLM

Reranking improves precision for:
- Financial tables
- Similar accounting terminology
- Ambiguous phrases (e.g., "Total term debt" vs "Total term debt principal")

---

## Step 3 — LLM Generation

Model used:
- `mistralai/Mistral-7B-Instruct-v0.3`

Prompt engineering:
- Few shot (3 shot with examples)

---

# Future work:
1. Clean verbosity by making LLM summarize the texts of the chunks
2. Chunk tables separately instead of relying on recursive character split
3. Batch inferencing
4. Guardrails
---
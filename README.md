# Internal Data Engineering Knowledge RAG Pipeline
## 📌 Overview

This project is a Retrieval-Augmented Generation (RAG) pipeline designed to make internal Data Engineering documentation, Python scripts, and dbt models easily accessible through a Large Language Model (LLM).

By parsing and indexing internal resources, the system allows team members to query the company’s technical knowledge base in natural language, improving onboarding, troubleshooting, and knowledge sharing.

## ⚙️ Features

Document Parsing – Extracts content from internal documents (e.g., markdown, Word, Google Docs exports).

Code Indexing – Parses documented Python scripts and dbt models for searchability.

RAG Architecture – Combines a vector search index with an LLM for context-aware responses.

Multi-Source Support – Handles a mix of text documents, codebases, and dbt SQL models.

Searchable Knowledge Base – Allows semantic search across all indexed content.

## 🛠️ Tech Stack

Programming Language: Python

Vector Store: FAISS

Embedding Model: HuggingFace embeddings

LLM: mistral-7b

Parsing: LangChain document loaders & custom parsers

Orchestration: Python scripts


## 📂 Project Structure

├── data/                     # Raw exported documents & scripts
├── src/
│   ├── embeddings.py         # Parses and indexes data
│   ├── query_pipeline.py     # Queries the vector DB via LLM
└── README.md    


## 🚀 How It Works

Ingestion & Parsing – Reads internal docs, Python scripts, and dbt SQL files, then cleans and normalizes them.

Embedding & Indexing – Converts content into vector embeddings and stores them in a vector database.

Query Processing – Accepts a natural language question, retrieves the most relevant indexed content, and sends it to the LLM.

Context-Aware Answering – The LLM uses retrieved context to generate accurate, source-backed responses.

## 📈 Use Cases

Speeding up onboarding for new data engineers

Reducing manual searching through internal docs

Centralizing tribal knowledge into one searchable system

Supporting LLM-powered assistants for technical Q&A

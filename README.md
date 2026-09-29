
# 🌱 AgriLLM — Crop Disease Knowledge Assistant

A domain-specific AI assistant for answering crop disease questions using a **Retrieval-Augmented Generation (RAG)** pipeline combined with **LoRA fine-tuning**.

The project uses a small open-source language model, a vector search system, and a curated agricultural knowledge base to reduce hallucinations and provide grounded answers.

---

## 🚀 Project Overview

The goal of this project was to explore how Large Language Models can be adapted for a specific domain.

The project demonstrates:

- Large Language Model inference
- Retrieval-Augmented Generation (RAG)
- Semantic embeddings
- FAISS vector search
- LoRA parameter-efficient fine-tuning
- Retrieval evaluation using Recall@3
- Streamlit application development

---

## 🏗️ Architecture

```text
                    User Question
                         │
                         ▼
              Sentence Transformer
                    Embeddings
                         │
                         ▼
                  FAISS Vector DB
                         │
                    Top-K Retrieval
                         │
                         ▼
                Relevant Documents
                         │
                         ▼
                Context Construction
                         │
                         ▼
                 Qwen3-0.6B + LoRA
                         │
                         ▼
                       Answer

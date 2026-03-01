
---

# ✅ 2️⃣ Design Report.md (1 Page Concept Note)

This should be concise, technical, and justify decisions.

---

```markdown
# Design Report – Financial RAG System

## Objective

Build a Retrieval-Augmented Generation (RAG) system that answers complex financial and legal questions using Apple’s 2024 10-K and Tesla’s 2023 10-K filings.

The system must:
- Use only retrieved context
- Cite document sources
- Refuse out-of-scope questions
- Use open-source models only

---

## 1. Document Ingestion & Chunking Strategy

PDFs were loaded using `PyPDFLoader`.

Chunking approach:
- Primary split: RecursiveCharacterTextSplitter (1400 chars, 100 overlap)
- Secondary split: Regex-based splitting by `Item X` section headers

Rationale:
SEC filings are structured by Items (Item 1, Item 1A, Item 7, Item 8, etc.). Splitting by section headers improves semantic coherence and retrieval precision.

Metadata preserved:
- `document` (Apple 10-K / Tesla 10-K)
- `page` (original page number)

This enables accurate source citation.

---

## 2. Embeddings & Vector Store

Embedding model:
- `BAAI/bge-base-en-v1.5`
- Normalized embeddings

Vector store:
- FAISS

Reasoning:
BGE models perform strongly on semantic retrieval benchmarks and work efficiently in local environments.

---

## 3. Retrieval & Reranking

Pipeline:
1. Top-20 similarity retrieval
2. Cross-encoder reranking using `BAAI/bge-reranker-base`
3. Top-5 chunks sent to LLM

Why reranking?
Financial documents contain semantically similar phrases:
- “Total term debt”
- “Total term debt principal”

Bi-encoders may retrieve near matches.  
Cross-encoders significantly improve precision by scoring query–document pairs jointly.

---

## 4. Query Rewriting

A lightweight LLM-based query optimizer rewrites natural language questions into short keyword-style search queries.

Example:
"What was Apple's total revenue..."  
→ "Apple total revenue fiscal year ended September 28 2024 Item 8"

This improves retrieval alignment with SEC document phrasing.

---

## 5. LLM Selection

Model:
- `mistralai/Mistral-7B-Instruct-v0.3`

Reasons:
- Fully open-source
- Strong instruction-following capability
- Efficient inference
- Good performance for financial reasoning tasks

Prompt design enforces:
- Context-only answering
- Explicit refusal if answer not present
- No hallucinations
- One-sentence output

---

## 6. Out-of-Scope Handling

If:
- Answer not present in retrieved context
- Question requires future knowledge
- Question unrelated to documents

System responds:

"This question cannot be answered based on the provided documents."

This prevents hallucination and ensures reliability.

---

## 7. System Strengths

- Hybrid retrieval (bi-encoder + cross-encoder)
- Structured SEC-aware chunking
- Metadata-preserving citations
- Fully open-source stack
- Cloud-runnable notebook

---

## 8. Limitations

- Table extraction is text-based (no structured table parsing)
- Numeric reasoning depends on LLM interpretation
- Large model inference may require GPU for efficiency

---

## Conclusion

The system provides a robust, production-style financial RAG pipeline that balances retrieval precision, interpretability, and hallucination control while remaining fully open-source and reproducible.
import time
start = time.perf_counter()

print("Started")

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

APPLE_DOC = "https://s2.q4cdn.com/470004039/files/doc_earnings/2024/q4/filing/10-Q4-2024-As-Filed.pdf"
TESLA_DOC = "https://ir.tesla.com/_flysystem/s3/sec/000162828024002390/tsla-20231231-gen.pdf"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

apple_loader = PyPDFLoader(APPLE_DOC)
tesla_loader = PyPDFLoader(TESLA_DOC)

apple_doc = apple_loader.load()
tesla_doc = tesla_loader.load()

def add_metadata(docs, document_name):
  for doc in docs:
    doc.metadata['document'] = document_name
    doc.metadata['page'] = doc.metadata.get('page', None)
  return docs

apple_doc = add_metadata(apple_doc, "Apple 10-K")
tesla_doc = add_metadata(tesla_doc, "Tesla 10-K")


print("Extracted data from PDFs")

# Remove table of contents
apple_doc = [doc for doc in apple_doc if doc.metadata["page"] != 2]
tesla_doc = [doc for doc in tesla_doc if doc.metadata["page"] != 2]
all_docs = apple_doc + tesla_doc

import re
from langchain_core.documents import Document

def split_by_item_headers(docs):
    pattern = r"(\nItem\s+\d+[A-Z]?\.?.*?)\n"
    structured_docs = []

    for doc in docs:
        text = doc.page_content

        splits = re.split(pattern, text)

        # re.split keeps headers separately, so recombine
        for i in range(1, len(splits), 2):
            header = splits[i].strip()
            content = splits[i+1].strip() if i+1 < len(splits) else ""

            structured_docs.append(
                Document(
                    page_content=header + "\n" + content,
                    metadata=doc.metadata
                )
            )

    return structured_docs

apple_doc = split_by_item_headers(apple_doc)
tesla_doc = split_by_item_headers(tesla_doc)
all_docs = split_by_item_headers(all_docs)


# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunked_docs_apple = text_splitter.split_documents(apple_doc)
chunked_docs_tesla = text_splitter.split_documents(tesla_doc)
chunked_docs_combined = text_splitter.split_documents(all_docs)

# for doc in chunked_docs:
#    if (doc.metadata["page"] == 19) & ("Item 1B" in doc.page_content):
#       print(f"Page 20 content: {doc.page_content}")

print("Chunking completed")

# Embedding
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding completed")


# Put embeddings in vector db
from langchain_community.vectorstores import FAISS

vector_store_apple = FAISS.from_documents(
    chunked_docs_apple,
    embedding_model
)

retriever_apple = vector_store_apple.as_retriever(search_kwargs={"k":20})

print(retriever_apple.invoke("What was Apples total revenue for the fiscal year ended September 28, 2024?"))

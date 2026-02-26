import time
start = time.perf_counter()

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

all_docs = apple_doc + tesla_doc

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunked_docs = text_splitter.split_documents(all_docs)

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

from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(
    chunked_docs,
    embedding_model
)

retriever = vector_store.as_retriever(search_kwargs={"k":5})

# def build_prompt(query, docs):
#   context = "\n\n".join([
#       f"[Source: {doc.metadata['document']}, Page: {doc.metadata['page']}]\n{doc.page_content}"
#       for doc in docs
#   ])

#   prompt = f"""
#   You are a financial analyst assistant.

#   Use only the context below to answer to answer the question.
#   If the answer is not in the context, say "Not found in provided documents."

#   Context:
#   {context}

#   Question:
#   {query}

#   Answer:
#   """

#   return prompt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto" if device=="cuda" else None,
    dtype=torch.float32
)

generator = pipeline(
  "text-generation",
  model=model,
  tokenizer=tokenizer)

def format_prompt(query, context):
  return f"""<|system|>
  You are a financial analyst assistant.
  Answer using only the provided context.
  If answer is not found, say "This question cannot be answered based on the provided documents."
  <|user|>
  Context:
  {context}

  Question:
  {query}
  <|assistant|>
  """

def call_llm(prompt):
  output = generator(
      prompt,
      max_new_tokens=1000,
      temperature=0,
      do_sample=False)
  return output[0]["generated_text"][len(prompt):]

def answer_question(query):
  docs = retriever.invoke(query)

  context = "\n\n".join([
      f"[Source: {doc.metadata['document']}, Page: {doc.metadata['page']}]\n{doc.page_content}"
      for doc in docs
  ])

  prompt = format_prompt(query, context)

  answer = call_llm(prompt)

  return answer, docs

questions = [
{"question_id": 1, "question": "What was Apples total revenue for the fiscal year ended September 28, 2024?"},
{"question_id": 2, "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?"},
{"question_id": 3, "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"},
{"question_id": 4, "question": "On what date was Apples 10-K report for 2024 signed and filed with the SEC?"},
{"question_id": 5, "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"},
{"question_id": 6, "question": "What was Teslas total revenue for the year ended December 31, 2023?"},
{"question_id": 7, "question": "What percentage of Teslas total revenue in 2023 came from Automotive Sales (excluding Leasing)?"},
{"question_id": 8, "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?"},
{"question_id": 9, "question": "What types of vehicles does Tesla currently produce and deliver?"},
{"question_id": 10, "question": "What is the purpose of Teslas ’lease pass-through fund arrangements’?"},
{"question_id": 11, "question": "What is Teslas stock price forecast for 2025?"},
{"question_id": 12, "question": "Who is the CFO of Apple as of 2025?"},
{"question_id": 13, "question": "What color is Teslas headquarters painted?"}
]

# for i in range(len(questions)):
#   question = questions[i]['question']
#   answer, docs = answer_question(question)
#   print(f"Question: {question}")
#   print(f"Answer: {answer}")
#   print(f"Time elapsed: {time.perf_counter() - start:.2f} seconds")
#   print("="*80, "\n")

import json
results = []
for question in questions:
  question_id = question['question_id']
  question_text = question['question']

  answer, docs = answer_question(question_text)

  if "this question cannot be answered based on the provided documents" in answer.lower():
    answer_text = "This question cannot be answered based on the provided documents"
    sources = []
  else:
    answer_text = answer.strip()
    sources = [doc.metadata['document'] for doc in docs]

  results.append({
    "question_id": question_id,
    "answer": answer_text,
    "sources": sources
  })

print(f"Total time taken = {time.perf_counter()-start:.0f} seconds")

print(results)


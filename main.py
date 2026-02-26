import time
start = time.perf_counter()

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

# model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto" if device=="cuda" else None,
    dtype=torch.bfloat16
)

# generator = pipeline(
#   "text-generation",
#   model=model,
#   tokenizer=tokenizer)

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

# def call_llm(prompt):
#   output = generator(
#       prompt,
#       max_new_tokens=1000,
#       temperature=0,
#       do_sample=False)
#   return output[0]["generated_text"][len(prompt):]


import torch

def call_llm(prompt, max_new_tokens=300):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

# def answer_question(query):
#   docs = retriever.invoke(query)

#   context = "\n\n".join([
#       f"[Source: {doc.metadata['document']}, Page: {doc.metadata['page']}]\n{doc.page_content}"
#       for doc in docs
#   ])

#   prompt = format_prompt(query, context)

#   answer = call_llm(prompt)

#   return answer, docs

def answer_question(query: str) -> dict:
  """
  Answers a question using the RAG pipeline.
  Args:
  query (str): The user question about Apple or Tesla 10-K filings.
  
  Returns:
  dict: {
  "answer": "Answer text or 'This question cannot be answered based on the provided documents.'",
  "sources": ["Apple 10-K", "Item 8", "p. 28"] # Empty list if refused}
  """
  # Your RAG logic here
  docs = retriever.invoke(query)

  # print(docs[0].metadata)
  sources =[]
  for doc in docs:
    sources.append((doc.metadata['document'], "Item ??", f"p. {doc.metadata['page']}"))

  context = "\n\n".join([
      f"[Source: {doc.metadata['document']}, Page: {doc.metadata['page']}]\n{doc.page_content}"
      for doc in docs
  ])

  prompt = format_prompt(query, context)

  answer = call_llm(prompt)
  return {"answer": answer, "sources": sources}

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

  response = answer_question(question_text)
  response['question_id'] = question_id
  results.append(response)

  # if "this question cannot be answered based on the provided documents" in answer.lower():
  #   answer_text = "This question cannot be answered based on the provided documents"
  #   sources = []
  # else:
  #   answer_text = answer.strip()
  #   sources = [doc.metadata['document'] for doc in docs]

  # results.append({
  #   "question_id": question_id,
  #   "answer": answer_text,
  #   "sources": sources
  # })

print(f"Total time taken = {time.perf_counter()-start:.0f} seconds")

# for i in results:
#   print(i["question_id"])
#   print(i["answer"])
#   print(i["sources"])
#   print("="*200, "\n")

# Evaluation
ground_truth = [{
  "question_id": 1, "answer": "$391,036 million"
  },
  {
    "question_id": 2, "answer": "15,115,823,000 shares"
  },
  {
    "question_id": 3, "answer": "$96,662 million"
  },
  {
    "question_id": 4, "answer": "November 1, 2024"
  },
  {
    "question_id": 5, "answer": "No. Checkmark indicates 'No' under Item 1B"
  },
  {
    "question_id": 6, "answer": "$96,773 million"
  },
  {
    "question_id": 7, "answer": "∼84% ($81,924M / $96,773M)"
  },
  {
    "question_id": 8, "answer": "Central to strategy, innovation, leadership; loss could disrupt"
  },
  {
    "question_id": 9, "answer": "Model S, Model 3, Model X, Model Y, Cybertruck"
  },
  {
    "question_id": 10, "answer": "Finance solar systems with investors; customers sign PPAs"
  },
  {
    "question_id": 11, "answer": "This question cannot be answered based on the provided documents"
  },
  {
    "question_id": 12, "answer": "This question cannot be answered based on the provided documents"
  },
  {
    "question_id": 13, "answer": "This question cannot be answered based on the provided documents"
  }]

## Semantic similarity using Cosine similarity of embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def semantic_similarity(prediction: str, ground_truth: str) -> float:
  """
  Computes cosine similarity between the LLM output and the actual answer
  """
  emb_pred = embedding_model.embed_query(prediction)
  emb_truth = embedding_model.embed_query(ground_truth)

  score = cosine_similarity([emb_pred], [emb_truth])[0][0]

  return float(score)

evaluation_result = []
for result in results:
  question_id = result["question_id"]
  prediction = result["answer"]

  truth = next(item["answer"] for item in ground_truth if item["question_id"]==question_id)

  sim_score = semantic_similarity(prediction, truth)

  evaluation_result.append({
    "question_id": question_id,
    "prediction": prediction,
    "ground_truth": truth,
    "semantic_similarity": sim_score
  })

import pandas as pd
print(pd.DataFrame(evaluation_result).to_string())

## LLM as the eveluator
def build_eval_prompt(question, ground_truth, prediction):
  return f"""
You are an expert evaluator for finance questions and answers.
Your task is to check whether the model output is correct or not.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Model Answer:
{prediction}

Instructions:
- If the answer is factually correct and semantically equivalent to the ground truth, return: CORRECT
- If the answer contradicts, hallucinates, or is incorrect, return: INCORRECT
- If the ground truth says the question cannot be answered and the model properly refuses, return: CORRECT
- Output only one word: CORRECT or INCORRECT
"""


def llm_judge(question, ground_truth, prediction):
  prompt = build_eval_prompt(question, ground_truth, prediction)

  # output = generator(
  #   prompt,
  #   max_new_tokens=10,
  #   temperature=0,
  #   do_sample=False
  # )

  # verdict = output[0]["generated_text"][len(prompt):].strip()
  verdict = call_llm(prompt, max_new_tokens=10)

  return verdict.strip()

llm_eval_results = []
for result in results:
  question_id = result["question_id"]
  prediction = result["answer"]

  question_text = next(q["question"] for q in questions if q["question_id"]==question_id)
  ground_truth_answer = next(gt["answer"] for gt in ground_truth if gt["question_id"]==question_id)

  verdict = llm_judge(question_text, ground_truth_answer, prediction)

  llm_eval_results.append({
    "question_id": question_id,
    "verdict": verdict
  })

print(pd.DataFrame(llm_eval_results).to_string())

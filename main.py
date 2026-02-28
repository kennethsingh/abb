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
all_docs = apple_doc + tesla_doc

print("Extracted data from PDFs")


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
vector_store_tesla = FAISS.from_documents(
    chunked_docs_tesla,
    embedding_model
)
vector_store_combined = FAISS.from_documents(
    chunked_docs_combined,
    embedding_model
)

retriever_apple = vector_store_apple.as_retriever(search_kwargs={"k":20})
retriever_tesla = vector_store_tesla.as_retriever(search_kwargs={"k":20})
retriever_combined = vector_store_combined.as_retriever(search_kwargs={"k":20})



# docs = retriever.invoke("unresolved staff comments")
# for doc in docs:
#    print(f"Retrieved Page: {doc.metadata['page']}")

# Reranking
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

reranker_model_id = "BAAI/bge-reranker-base"

rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_model_id)
rerank_model = AutoModelForSequenceClassification.from_pretrained(
   reranker_model_id,
   device_map="auto" if device=="cuda" else None,
   dtype=torch.bfloat16 if device=="cuda" else torch.float32
   )
rerank_model.eval()

def rerank_documents(query, docs, top_k=5):
   """
   Rerank retrieved documents using cross-encoder and return top_k documents
   """
   pairs = [(query, doc.page_content) for doc in docs]

   inputs = rerank_tokenizer(
      pairs,
      padding=True,
      truncation=True,
      return_tensors="pt",
      max_length=512
   ).to(rerank_model.device)

   with torch.no_grad():
      outputs = rerank_model(**inputs)
      scores = outputs.logits.squeeze(-1)

   ranked_indices = torch.argsort(scores, descending=True)
   
   reranked_docs = [docs[i] for i in ranked_indices[:top_k]]

   return reranked_docs





import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# model_id = "Qwen/Qwen2.5-0.5B-Instruct"
# model_id = "openai/gpt-oss-120b"
# model_id = "Qwen/Qwen2.5-7B-Instruct"
# model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# model_id = "deepseek-ai/DeepSeek-R1-Zero"
# model_id = "Qwen/Qwen3-14B"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

print(f"Using Moodel: {model_id}")

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
  return f"""
  You are a financial analyst assistant.
  Answer using only the provided context.
  If answer is not found, say "This question cannot be answered based on the provided documents."
  Finally, output only the answer in one sentence.

  Example 1:
  Question: In which year was Tesla founder?
  Answer: Tesla was founded in the year 2003.

  Example 2:
  Question: Where is Apple's headquarters?
  Answer: Apple headquarters is in USA.

  Now answer this question:
  Context:
  {context}

  Question:
  {query}

  Answer:
  """


import torch

def call_answer_llm(prompt, max_new_tokens=50):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # do_sample=False
            temperature=0.1
        )

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )



def answer_question(query: str, company: str) -> dict:
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
  if company == "apple":
     docs = retriever_apple.invoke(query)
  elif company == "tesla":
     docs = retriever_tesla.invoke(query)
  else:
     docs = retriever_combined.invoke(query)

  docs = rerank_documents(query, docs, top_k=5)

  # print(docs[0].metadata)
  sources =[]
  for doc in docs:
    sources.append((doc.metadata['document'], "Item ??", f"p. {int(doc.metadata['page'])+1}"))
    

  context = "\n\n".join([
      f"[Source: {doc.metadata['document']}, Page: {int(doc.metadata['page'])+1}]\n{doc.page_content}"
      for doc in docs
  ])

  prompt = format_prompt(query, context)

  if "item 1b" in prompt.lower():
     print(f"CHECK METADATA: {prompt}")

  answer = call_answer_llm(prompt)
  return {"answer": answer, "sources": sources}

print("Question answer function created")


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

# ==================================================
# REPHRASE THE QUESTIONS FOR BETTER SEMANTIC MATCHING WITH THE DOC CHUNKS
# ==================================================
rephrase_model_id = "mistralai/Mistral-7B-Instruct-v0.3"

rephrase_tokenizer = AutoTokenizer.from_pretrained(rephrase_model_id)

rephrase_model = AutoModelForCausalLM.from_pretrained(
    rephrase_model_id,
    device_map="auto" if device=="cuda" else None,
    dtype=torch.bfloat16
)

def build_rephrase_prompt(question):
  return f"""
You are a retrieval query optimizer for SEC 10-K filings.

Your task is to convert a natural language question into a SHORT keyword search query.

Rules:
- DO NOT answer the question.
- DO NOT provide explanations.
- Focus on section titles or core financial terms.
- Remove conversational phrasing.

Now rewrite this:

Question: {question}
Search query:
"""

def call_rephrase_llm(prompt, max_new_tokens=50):
    inputs = rephrase_tokenizer(
        prompt,
        return_tensors="pt"
    ).to(rephrase_model.device)

    with torch.no_grad():
        output = rephrase_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
            # temperature=0.5
        )

    return rephrase_tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

rephrased_questions = []
for question in questions:
   question_text = question["question"]
   question_prompt = build_rephrase_prompt(question_text)
   rephrased_question = call_rephrase_llm(question_prompt)
   question["rephrased_question"] = rephrased_question
   rephrased_questions.append(question)

# print(f"Rephrased Questions: {rephrased_questions}")



print("Getting answers for the 13 questions")

import json
results = []
for question in rephrased_questions:
  question_id = question['question_id']
  question_text = question['question']
  rephrased_question_text = question['rephrased_question']

  if "apple" in question_text.lower():
     response = answer_question(query=question_text, company="apple")
     #  response = answer_question(rephrased_question_text)
  elif "tesla" in question_text.lower():
     response = answer_question(query=question_text, company="tesla")
     #  response = answer_question(rephrased_question_text)
  else:
     response = answer_question(query=question_text, company="combined")
  
  response['question_id'] = question_id
  results.append(response)



print(f"Total time taken = {time.perf_counter()-start:.0f} seconds")



print("Answers received")

print("Evaluating output")
# Evaluation
ground_truth = [{"question_id": 1, "answer": "$391,036 million"},
                {"question_id": 2, "answer": "15,115,823,000 shares"},
                {"question_id": 3, "answer": "$96,662 million"},
                {"question_id": 4, "answer": "November 1, 2024"},
                {"question_id": 5, "answer": "No. Checkmark indicates 'No' under Item 1B"},
                {"question_id": 6, "answer": "$96,773 million"},
                {"question_id": 7, "answer": "∼84% ($81,924M / $96,773M)"},
                {"question_id": 8, "answer": "Central to strategy, innovation, leadership; loss could disrupt"},
                {"question_id": 9, "answer": "Model S, Model 3, Model X, Model Y, Cybertruck"},
                {"question_id": 10, "answer": "Finance solar systems with investors; customers sign PPAs"},
                {"question_id": 11, "answer": "This question cannot be answered based on the provided documents"},
                {"question_id": 12, "answer": "This question cannot be answered based on the provided documents"},
                {"question_id": 13, "answer": "This question cannot be answered based on the provided documents"}]

# Compare prediction vs ground truth
truth_vs_prediction = []
for result in results:
  question_id = result["question_id"]
  prediction = result["answer"]
  
  sources = result["sources"]
  
  question_text = next(q["question"] for q in rephrased_questions if q["question_id"]==question_id)
  rephrased_question_text = next(q["rephrased_question"] for q in rephrased_questions if q["question_id"]==question_id)
  ground_truth_answer = next(gt["answer"] for gt in ground_truth if gt["question_id"]==question_id)
  truth_vs_prediction.append({
    "question_id": question_id,
    "question": question_text,
    # "rephrased_question": rephrased_question_text,
    "ground_truth": ground_truth_answer,
    "prediction": prediction,
    "sources": sources
  })

truth_vs_prediction = pd.DataFrame(truth_vs_prediction)
truth_vs_prediction.to_csv("truth_vs_prediction.csv", index = False)


## Semantic similarity using Cosine similarity of embeddings
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# def semantic_similarity(prediction: str, ground_truth: str) -> float:
#   """
#   Computes cosine similarity between the LLM output and the actual answer
#   """
#   emb_pred = embedding_model.embed_query(prediction)
#   emb_truth = embedding_model.embed_query(ground_truth)

#   score = cosine_similarity([emb_pred], [emb_truth])[0][0]

#   return float(score)

# evaluation_result = []
# for result in results:
#   question_id = result["question_id"]
#   prediction = result["answer"]
#   sources = result["sources"]

#   truth = next(item["answer"] for item in ground_truth if item["question_id"]==question_id)

#   sim_score = semantic_similarity(prediction, truth)

#   evaluation_result.append({
#     "question_id": question_id,
#     "prediction": prediction,
#     "ground_truth": truth,
#     "sources": sources,
#     "semantic_similarity": sim_score
#   })


# import pandas as pd
# # print(pd.DataFrame(evaluation_result).to_string())

# # Save semantice similarity evaluation dataframe
# semantic_evaluation_df = pd.DataFrame(evaluation_result)
# semantic_evaluation_df.to_csv("semantic_evaluation_df.csv", index=False)

## LLM as the eveluator
# eval_model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_id)

# eval_model = AutoModelForCausalLM.from_pretrained(
#     eval_model_id,
#     device_map="auto" if device=="cuda" else None,
#     dtype=torch.bfloat16
# )

# def build_eval_prompt(question, ground_truth, prediction):
#   return f"""
# You are an expert evaluator for finance questions and answers.
# Your task is to check whether the model output is correct or not.

# Question:
# {question}

# Ground Truth Answer:
# {ground_truth}

# Model Answer:
# {prediction}

# Rules:
#  - Return only one word: Either CORRECT or INCORRECT

# Instructions:
# - If the answer is factually correct and semantically equivalent to the ground truth, return: CORRECT
# - If the answer contradicts, hallucinates, or is incorrect, return: INCORRECT
# - If the ground truth says the question cannot be answered and the model properly refuses, return: CORRECT

# Answer:
# """

# def call_eval_llm(prompt, max_new_tokens=10):
#     inputs = eval_tokenizer(
#         prompt,
#         return_tensors="pt"
#     ).to(eval_model.device)

#     with torch.no_grad():
#         output = eval_model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False
#             # temperature=0.5
#         )

#     return eval_tokenizer.decode(
#         output[0][inputs["input_ids"].shape[1]:],
#         skip_special_tokens=True
#     )

# def llm_judge(question, ground_truth, prediction):
#   prompt = build_eval_prompt(question, ground_truth, prediction)

#   verdict = call_eval_llm(prompt, max_new_tokens=10)

#   return verdict.strip()

# llm_eval_results = []
# for result in results:
#   question_id = result["question_id"]
#   prediction = result["answer"]
  
#   sources = result["sources"]
  

#   question_text = next(q["question"] for q in rephrased_questions if q["question_id"]==question_id)
#   rephrased_question_text = next(q["rephrased_question"] for q in rephrased_questions if q["question_id"]==question_id)
#   ground_truth_answer = next(gt["answer"] for gt in ground_truth if gt["question_id"]==question_id)

#   verdict = llm_judge(rephrased_question_text, ground_truth_answer, prediction)

#   llm_eval_results.append({
#     "question_id": question_id,
#     "question": question_text,
#     "rephrased_question": rephrased_question_text,
#     "ground_truth": ground_truth_answer,
#     "prediction": prediction,
#     "sources": sources,
#     "verdict": verdict
#   })

# print(pd.DataFrame(llm_eval_results).to_string())

# # for i in llm_eval_results:
# #   print(f"Question ID: {i["question_id"]}")
# #   print(f"Verdict: {i['verdict']}")
# #   print("="*200, "\n\n")

# # Save llm evaluation dataframe
# llm_evaluation_df = pd.DataFrame(llm_eval_results)
# llm_evaluation_df.to_csv("llm_evaluation_df.csv", index=False)
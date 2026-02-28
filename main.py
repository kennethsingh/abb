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
# all_docs = apple_doc + tesla_doc

print("Extracted data from PDFs")

# Remove table of contents
apple_doc = [doc for doc in apple_doc if doc.metadata["page"] != 2]
tesla_doc = [doc for doc in tesla_doc if doc.metadata["page"] != 2]
all_docs = apple_doc + tesla_doc


import re

def reconstruct_paragraphs(text):
    lines = text.split("\n")
    paragraphs = []
    current = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if not current:
            current = line
            continue

        # If previous line does not end with punctuation → likely wrapped
        if not re.search(r"[.:;!?]$", current):
            current += " " + line
        # If next line starts lowercase → continuation
        elif line and line[0].islower():
            current += " " + line
        else:
            paragraphs.append(current.strip())
            current = line

    if current:
        paragraphs.append(current.strip())

    return paragraphs

from langchain_core.documents import Document

def chunk_docs_by_paragraph(docs, min_chunk_size=400):
    chunked_docs = []

    for doc in docs:
        paragraphs = reconstruct_paragraphs(doc.page_content)

        buffer = ""
        for para in paragraphs:
            # Skip tiny noise
            if len(para.strip()) < 20:
                continue

            if len(buffer) + len(para) < min_chunk_size:
                buffer += " " + para
            else:
                chunked_docs.append(
                    Document(
                        page_content=buffer.strip(),
                        metadata=doc.metadata.copy()
                    )
                )
                buffer = para

        if buffer:
            chunked_docs.append(
                Document(
                    page_content=buffer.strip(),
                    metadata=doc.metadata.copy()
                )
            )

    return chunked_docs


chunked_docs_apple = chunk_docs_by_paragraph(apple_doc)
chunked_docs_tesla = chunk_docs_by_paragraph(tesla_doc)
chunked_docs_combined = chunk_docs_by_paragraph(all_docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunked_docs_apple = text_splitter.split_documents(chunked_docs_apple)
chunked_docs_tesla = text_splitter.split_documents(chunked_docs_apple)
chunked_docs_combined = text_splitter.split_documents(chunked_docs_apple)



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

  docs = rerank_documents(query, docs, top_k=10)

  # print(docs[0].metadata)
  sources =[]
  for doc in docs:
    sources.append((doc.metadata['document'], "Item ??", f"p. {int(doc.metadata['page'])+1}"))
    

  context = "\n\n".join([
     f"""
     ---START CHUNK---
     [Source: {doc.metadata['document']}, Page: {int(doc.metadata['page'])+1}]\n{doc.page_content}
     ---END CHUNK---"""
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
ground_truth = [{"question_id": 1, "answer": "$391,036 million", "source": "Apple 10-K, Item 8, p. 282"},
                {"question_id": 2, "answer": "15,115,823,000 shares", "source": "Apple 10-K, first paragraph"},
                {"question_id": 3, "answer": "$96,662 million", "source": "Apple 10-K, Item 8, Note 9, p. 394"},
                {"question_id": 4, "answer": "November 1, 2024", "source": "Apple 10-K, Signature page"},
                {"question_id": 5, "answer": "No. Checkmark indicates 'No' under Item 1B", "source": "Apple 10-K, Item 1B, p. 176"},
                {"question_id": 6, "answer": "$96,773 million", "source": "Tesla 10-K, Item 7"},
                {"question_id": 7, "answer": "∼84% ($81,924M / $96,773M)", "source": "Tesla 10-K, Item 7"},
                {"question_id": 8, "answer": "Central to strategy, innovation, leadership; loss could disrupt", "source": "Tesla 10-K, Item 1A"},
                {"question_id": 9, "answer": "Model S, Model 3, Model X, Model Y, Cybertruck", "source": "Tesla 10-K, Item 1"},
                {"question_id": 10, "answer": "Finance solar systems with investors; customers sign PPAs", "source": "Tesla 10-K, Item 7"},
                {"question_id": 11, "answer": "This question cannot be answered based on the provided documents", "source": "N/A"},
                {"question_id": 12, "answer": "This question cannot be answered based on the provided documents", "source": "N/A"},
                {"question_id": 13, "answer": "This question cannot be answered based on the provided documents", "source": "N/A"}]

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


# def combine_pages_with_markers(docs):
#    docs = sorted(docs, key=lambda x: x.metadata["page"])

#    full_text = ""
#    for doc in docs:
#       page_number = doc.metadata["page"] + 1
#       full_text += f"\n\n<<PAGE: {page_number}>>\n"
#       full_text += doc.page_content
    
#    return full_text

# apple_doc = combine_pages_with_markers(apple_doc)
# tesla_doc = combine_pages_with_markers(tesla_doc)

# import re
# from langchain_core.documents import Document

# def split_by_item_headers(full_text, document_name):
#    pattern = r"\n(?=Item\s+\d+[A-Z]?\.)"
#    sections = re.split(pattern, full_text, flags=re.IGNORECASE)

#    structured_docs = []

#    for section in sections:
#       section = section.strip()
#       if not section:
#          continue
      
#       # Extract item number
#       match = re.match(r"Item\s+(\d+[A-Z]?)\.", section, flags=re.IGNORECASE)
#       item_number = match.group(1) if match else None

#       # Extract page numbers from each section
#       pages = re.findall(r"<<PAGE: (\d+)>>", section)
#       pages = [int(p) for p in pages]

#       if pages:
#          page_start = min(pages)
#          page_end = max(pages)
#       else:
#          page_start = None
#          page_end = None

#       # Remove page markers
#       clean_section = re.sub(r"<<PAGE: \d+>>", "", section)

#       structured_docs.append(Document(page_content=clean_section.strip(),
#                                       metadata={
#                                          "document": document_name,
#                                          "item": item_number,
#                                          "page_start": page_start,
#                                          "page_end": page_end
#                                       }))
      
#    return structured_docs

# apple_doc = split_by_item_headers(full_text=apple_doc, document_name="Apple 10-K")

# print(apple_doc[3])
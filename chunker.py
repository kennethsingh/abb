from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from langchain_core.documents import Document

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1400,
    chunk_overlap=100
)

def chunker(doc):
    chunked_docs_apple = text_splitter.split_documents(doc)
    # chunked_docs_tesla = text_splitter.split_documents(tesla_doc)

def split_by_item_headers(docs):
    pattern = r"(\nItem\s+\d+[A-Z]?\.?.*?)\n"
    structured_docs = []

    for doc in docs:
        text = doc.page_content
        splits = re.split(pattern, text)

        # If no Item header found → keep full page
        if len(splits) <= 1:
            structured_docs.append(
                Document(
                    page_content=text.strip(),
                    metadata=doc.metadata.copy()
                )
            )
            continue

        # If headers found → recombine properly
        for i in range(1, len(splits), 2):
            header = splits[i].strip()
            content = splits[i+1].strip() if i+1 < len(splits) else ""

            structured_docs.append(
                Document(
                    page_content=header + "\n" + content,
                    metadata=doc.metadata.copy()
                )
            )

    return structured_docs

chunked_docs_apple = split_by_item_headers(chunked_docs_apple)
chunked_docs_tesla = split_by_item_headers(chunked_docs_tesla)
# chunked_docs_combined = split_by_item_headers(chunked_docs_combined)

# Enrich short chunks by adding generic keywords from the doc
import numpy as np

character_count_apple = []
for doc in chunked_docs_apple:
    character_count_apple.append(len(doc.page_content))

for i in range(len(chunked_docs_apple)):
    if len(chunked_docs_apple[i].page_content) < np.percentile(character_count_apple, 25):
        chunked_docs_apple[i].page_content = "Apple SEC 10-K report: " + chunked_docs_apple[i].page_content

character_count_tesla = []
for doc in chunked_docs_tesla:
    character_count_tesla.append(len(doc.page_content))

for i in range(len(chunked_docs_tesla)):
    if len(chunked_docs_tesla[i].page_content) < np.percentile(character_count_tesla, 25):
        chunked_docs_tesla[i].page_content = "Tesla SEC 10-K report: " + chunked_docs_tesla[i].page_content

print("Chunking completed")

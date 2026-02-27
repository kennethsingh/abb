from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================
# EXTRACT DATA FROM PDFs
# ============================================
APPLE_DOC = "https://s2.q4cdn.com/470004039/files/doc_earnings/2024/q4/filing/10-Q4-2024-As-Filed.pdf"
TESLA_DOC = "https://ir.tesla.com/_flysystem/s3/sec/000162828024002390/tsla-20231231-gen.pdf"

def data_extract(doc: str, document_name: str):
    doc_loader = PyPDFLoader(doc)

    docs = doc_loader.load()

    for doc in docs:
        doc.metadata["document"] = document_name
        doc.metadata["page"] = doc.metadata.get("page", None)

    return docs

apple_doc = data_extract(APPLE_DOC, "Apple 10-K")
tesla_doc = data_extract(TESLA_DOC, "Tesla 10-K")

all_docs = apple_doc + tesla_doc

print(apple_doc[0])


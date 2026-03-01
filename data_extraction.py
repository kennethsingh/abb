from langchain_community.document_loaders import PyPDFLoader

def data_extraction(doc, document_name):
    doc_loader = PyPDFLoader(doc)

    docs = doc_loader.load()

    for doc in docs:
        doc.metadata['document'] = document_name
        doc.metadata['page'] = doc.metadata.get('page', None)

    # Remove table of contents
    docs = [doc for doc in docs if doc.metadata["page"] != 2]
    
    print("Extracted data from PDFs")
    return docs

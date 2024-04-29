from SmartSearch.documents import get_document_info, create_index
from langchain.embeddings import SentenceTransformerEmbeddings

def build(docs_path, index_name, index_path, documents, embeddings):
   create_index(documents, embeddings, index_path, index_name)

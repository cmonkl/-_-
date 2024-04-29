from langchain.document_loaders import TextLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
import os
import pickle
import rutokenizer
from langchain.retrievers import BM25Retriever
from spacy.lang.ru import Russian
from spacy_russian_tokenizer import RussianTokenizer, MERGE_PATTERNS


DOC2LOADER = {
    '.txt': (TextLoader, {"encoding": "utf8"}),
    '.xlsx': (UnstructuredExcelLoader, {}),
    '.docx':(UnstructuredWordDocumentLoader, {})
}

def process_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if len(line.strip()) > 2]
    text = "\n".join(lines).strip()
    if len(text) < 10:
        return None
    return text

def get_document_info(document_dir):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200, #150,
        chunk_overlap  = 70, #70,
        length_function = len,
    )
    documents = []
    for root, dirs, files in os.walk(document_dir):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            loader, kwargs = DOC2LOADER[file_ext]
            loader = loader(os.path.join(root, file), **kwargs)
            data = loader.load()
            for data_item in data:
                data_item.page_content = process_text(data_item.page_content)
            chunks = text_splitter.split_documents(data)
            documents.extend(chunks)
    return documents

def create_index(documents, embeddings, save_path, index_name):   
    """local_vectors = FAISS.load_local(index_name=index_name, embeddings=embeddings, folder_path=save_path) 
    paths = set([val.metadata['source'].split('/')[-1] for val in local_vectors.docstore._dict.values()])
    new_documents = [doc for doc in documents if doc.metadata['source'].split('/')[1] not in paths]
   
    if len(new_documents) > 0:
        local_vectors.add_documents(new_documents)
    print(paths)
    #paths = set([val.metadata['source'].split('/')[-1] for val in local_vectors.docstore._dict.values()])
    #new_vectors = FAISS.from_documents(new_documents, embeddings)
    
    #try:
    #    local_vectors.merge_from(new_vectors)  # AttributeError: 'IndexFlat' object has no attribute 'merge_from'
    #except Exception as e:
        # Create a new FAISS index with the same dimension as the original indexes
    #    new_index = FAISS.create_faiss_index(local_vectors.d)

        # Add vectors from both indexes to the new index
    #    new_index.add(np.vstack((local_vectors.reconstruct(i) for i in range(local_vectors.ntotal))))
    #    new_index.add(np.vstack((new_vectors.reconstruct(i) for i in range(new_vectors.ntotal))))

        # Create a new FAISS vectorstore with the merged index
    #    local_vectors = FAISS(new_index, embeddings)
    
    local_vectors.save_local(index_name=index_name, folder_path=save_path)
    """
    vectors = FAISS.from_documents(documents, embeddings)
    vectors.save_local(index_name=index_name, folder_path=save_path)

def tokenize(s: str) -> list[str]:
    t = rutokenizer.Tokenizer()
    t.load()
    return t.tokenize(s.lower())
    # nlp = Russian()
    # russian_tokenizer = RussianTokenizer(nlp, MERGE_PATTERNS)
    # nlp.add_pipe(russian_tokenizer, name='russian_tokenizer')
    # return nlp(s.lower())


def bm25(documents, k):
    bm25_retriever = BM25Retriever.from_documents(
    documents=documents,
    preprocess_func=tokenize,
    k=k,
    )
    return bm25_retriever
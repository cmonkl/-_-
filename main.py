from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from SmartSearch.documents import get_document_info, create_index, bm25
from SmartSearch.model import get_llm
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
import time
from langchain.vectorstores.utils import DistanceStrategy
from langchain.retrievers import EnsembleRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
import os
from SmartSearch.build_index import build


def get_qa_model(index_path, index_name, embedding,  bm25_retriever, llm, prompt_template):
    vectors = FAISS.load_local(index_name=index_name, embeddings=embedding, folder_path=index_path, 
                    distance_strategy=DistanceStrategy.COSINE)
    vectors_retriever = vectors.as_retriever(search_type="mmr", 
                                        search_kwargs={"k": 5, 
                                        "score_threshold":0.3}) #search_kwargs={"k": 3}, 
    
    ensemble_retriever = EnsembleRetriever(
    retrievers=[vectors_retriever, bm25_retriever],
    weights=[0.6, 0.4],
    )

    qa_model = RetrievalQA.from_chain_type(
      llm=llm, 
      chain_type="stuff",
      retriever=ensemble_retriever, #vectors_retriever,
      return_source_documents=True,
      chain_type_kwargs={
        "prompt": PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        ),
      }
    )
    return qa_model

def get_answer(query, qa_model):
    generation = qa_model(query)
    answer = generation['result'] if len(generation['source_documents']) > 0 else "No relevant information found"
    return answer, generation


def calculate_cosine_similarity(answer, contexts):
    # Combine the answer and contexts for TF-IDF vectorization
    documents = [answer] + contexts

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF to our documents
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate cosine similarity of the answer with each context
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    return cosine_similarities.flatten()

def login():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Check credentials (a simple example, you might want to use a more secure method)
        if username == "demo" and password == "password":
            st.success("Вход выполнен успешно")
            st.session_state.user_logged_in = True
            st.session_state.username = username
            #st.empty()
            st.rerun()
            return True
        else:
            st.error("Неверные данные")
    return False

def question_answering():
    DOCS_PATH = 'docs'
    INDEX_PATH = 'faiss_index'
    INDEX_NAME  = 'index1'
    adapt_model_name = "/home/admin/model/smart_search/saiga_mistral_7b_lora"
    base_model_name = "/home/admin/model/smart_search/Mistral-7B-OpenOrca"
    prompt_template = "user: Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. Используя контекст: {context}, ответь на вопрос: {question}. Если ответа на вопрос в контексте нет, напиши, что ответа нет. \nbot: Вот ответ на ваш вопрос:"

    st.sidebar.title("Загрузка файлов")
    # Initialize a session state variable for button press
    if 'button_pressed' not in st.session_state:
        st.session_state['button_pressed'] = False
    uploaded_files = st.sidebar.file_uploader("Выбрать файл", accept_multiple_files=True)

    # Button to indicate that file uploading is complete
    if st.sidebar.button('Закончить загрузку') and st.session_state['button_pressed'] == False:
        if uploaded_files:    
            st.session_state['button_pressed'] = True

            st.sidebar.write("Загруженные файлы:")
            for uploaded_file in uploaded_files:
                st.sidebar.write(uploaded_file.name)
                with open(os.path.join("docs",uploaded_file.name),"wb") as f: 
                    f.write(uploaded_file.getbuffer())   
            
            documents = get_document_info(DOCS_PATH)
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            create_index(documents, embeddings, INDEX_PATH, INDEX_NAME)
            st.session_state['index'] = FAISS.load_local(index_name=INDEX_NAME,
                                    embeddings=embeddings,
                                    folder_path=INDEX_PATH, 
                    distance_strategy=DistanceStrategy.COSINE)
            bm25_retriever = bm25(documents, k=5)
          
            llm = get_llm(adapt_model_name, base_model_name)
            st.session_state['rag_model'] = get_qa_model(INDEX_PATH, INDEX_NAME, embeddings,
                                                          bm25_retriever, llm, prompt_template)#
            #get_qa_model(st.session_state['index'], llm)
        else:
            st.sidebar.write("Файлы еще не были загружены.")

    
    if st.session_state['button_pressed']:
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Введите вопрос"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                answer, generation = get_answer(prompt, st.session_state['rag_model'])
                parsed_answer = answer.split("Вот ответ на ваш вопрос:")[1].strip()
                if "bot:" in parsed_answer:
                    parsed_answer = parsed_answer.split("bot:")[0].strip()
                similarity_answer_contexts = calculate_cosine_similarity(answer, [doc.page_content for doc in generation['source_documents']])
                print("Similarity:", similarity_answer_contexts)
                threshold = 0.2
                if all(value < threshold for value in similarity_answer_contexts):
                    parsed_answer = "К сожалению, ответа на этот вопрос нет в документах."
                message_placeholder.markdown(parsed_answer)
            st.session_state.messages.append({"role": "assistant", "content": parsed_answer})


    if len(uploaded_files) == 0:    
        st.session_state['button_pressed'] = False
        st.session_state.messages = []

    print(st.session_state['button_pressed'])
          

def main():
    st.set_page_config(page_title='Вопросно-ответная система')
    st.title('Чат-бот по документам')

    # Check if the user is logged in
    if 'user_logged_in' not in st.session_state:
        st.session_state.user_logged_in = False

    if not st.session_state.user_logged_in:
        status = login()
        st.sidebar.warning("Пожалуйста, пройдите авторизацию")
    else:
        question_answering()

if __name__ == '__main__':
    main()



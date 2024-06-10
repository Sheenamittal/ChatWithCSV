import streamlit as st
import time
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

@st.cache(allow_output_mutation=True)
def load_model(data):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    print("Embedding done")

    db = FAISS.from_documents(data, embeddings)
    print("db done")
    db.save_local(DB_FAISS_PATH)
    print("db saved")

    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.0,
        context_length=1024
    )
    print("LLM loaded")

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
    print("chain made")

    return embeddings, chain

def split_long_input(input_text, max_length=1024):
    chunks = [input_text[i:i+max_length] for i in range(0, len(input_text), max_length)]
    return chunks

def conversational_chat(chain, query, history, max_length=1024):
    st.write("Processing answer...")
    start_time = time.time()
    query_chunks = split_long_input(query, max_length)

    result = {"answer": ""}
    for chunk in query_chunks:
        chunk_result = chain({"question": chunk, "chat_history": history})
        result["answer"] += chunk_result["answer"]

    end_time = time.time()
    processing_time = end_time - start_time
    st.write(f"Answer processed successfully! Time taken: {processing_time:.2f} seconds")

    history.append((query, result["answer"]))
    return result["answer"]

st.title("Chat with CSV 🦜")

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    st.write("Data is being loaded...")

    start_time = time.time()
    file_path = f"./temp/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    embeddings, chain = load_model(data)

    end_time = time.time()
    loading_time = end_time - start_time

    st.write(f"Model and data loaded successfully! Initial loading time: {loading_time:.2f} seconds")

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name]  

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your CSV data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            # Limit input to fit within the model's context length
            user_input = user_input[:1024]
            output = conversational_chat(chain, user_input, st.session_state['history'])
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                st.write(f"{st.session_state['past'][i]}: {st.session_state['generated'][i]}")
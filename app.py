import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.retrieval import retrieval
from src.loading import loader
from src.embedding import embedding,VectorStore
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

DB_PATH = r"D:\mlproject18\Database\vector_db"
DATA_DIR = r"D:\mlproject18\data"
SQLITE_URL = "sqlite:///D:/mlproject18/Database/chat_history.db"

os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def process(files_uploaded):
    embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db=Chroma(
        collection_name="clinical_database",
        embedding_function=embedding_model,
        persist_directory=DB_PATH
    )

    saved_file_paths=[]
    for files in files_uploaded:
        file_path = os.path.join(DATA_DIR, files.name)
        saved_file_paths.append(file_path)
        with open(file_path, "wb") as f:
            f.write(files.getbuffer())

        try:
            load_obj=loader(path=DATA_DIR)
            docs=load_obj.loading()
            texts=load_obj.chunking(docs)

            vector_db.add_documents(texts)
            for file_path in saved_file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)

        except Exception as e:
            for file_path in saved_file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)

            return False, f"Error processing {files.name}: {e}"

    return True,len(texts)

st.title("CliniBot")
st.markdown("Modern clinical assistant to help you with your any clinical queries regarding dosages, safety, and contraindications.")


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

retrieve_obj=retrieval()
chat_history_obj = retrieve_obj.session_history(st.session_state.session_id)

if "messages" not in st.session_state:
    st.session_state.messages = []
    for msg in chat_history_obj.messages:
        role = "user" if msg.type == "human" else "assistant"
        st.session_state.messages.append({"role": role, "content": msg.content})

with st.sidebar:
    st.header("Document Ingestion")
    st.markdown("Upload any number of **PDF** or **XML** clinical files. They will be parsed, embedded, and indexed immediately.")
    
    uploaded_files = st.file_uploader(
        "Choose clinical files", 
        type=["pdf", "xml"], 
        accept_multiple_files=True
    )

    if st.button("Process & Ingest Files", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing files, building embeddings, and updating vector DB..."):
                success, result = process(uploaded_files)
                if success:
                    st.success(f"Successfully ingested {result} chunks into vector store!")
                else:
                    st.error(result)
        else:
            st.warning("Please upload at least one PDF or XML file first.")

    st.markdown("---")
    st.header("Session Control")
    st.text(f"Session ID:\n{st.session_state.session_id}")
    if st.button("🔄 Clear & New Session", use_container_width=True):
        st.session_state.session_id=str(uuid.uuid4())
        chat_history_obj.clear()
        st.session_state.messages=[]
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a clinical query regarding dosages, safety, or contraindications..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            wrapped_chain=retrieve_obj.model_initialize(input_text=user_input, session_id=st.session_state.session_id) 
            def response_generator():
                full_response=""
                for chunk in wrapped_chain.stream(
                    {"input": user_input},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                ):
                    if hasattr(chunk, "content"):
                        text = chunk.content
                    else:
                        text = str(chunk)
                    full_response += text
                    yield text
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            st.write_stream(response_generator())
        except Exception as e:
            st.error(f"Execution error: {e}")
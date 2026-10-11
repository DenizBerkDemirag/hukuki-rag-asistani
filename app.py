import os
import uuid
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage

from graph_agent import legal_agent_app

# Sayfa Ayarı
st.set_page_config(page_title="Hukuki Asistan", page_icon="⚖️", layout="centered")

# Oturum Durumu
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Yan Panel: Belge Yönetimi
with st.sidebar:
    st.title("📄 Belgeler")
    uploaded_file = st.file_uploader("PDF Yükle", type=["pdf"], label_visibility="collapsed")

    if uploaded_file and st.button("Belgeyi İndeksle", use_container_width=True):
        with st.spinner("İndeksleniyor..."):
            os.makedirs("data", exist_ok=True)
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            docs = PyPDFLoader(file_path).load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
            Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
            st.success("Belge hazır!")

    if st.session_state.messages:
        st.divider()
        if st.button("Sohbeti Temizle", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

# Ana Ekran
st.title("⚖️ Hukuki Asistan")

# Sohbet Geçmişi
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Soru Girişi
if prompt := st.chat_input("Bir soru sorun..."):
    # Kullanıcı mesajı
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan yanıtı
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor..."):
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            inputs = {"question": prompt, "chat_history": history}
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            try:
                result = legal_agent_app.invoke(inputs, config=config)
                answer = result.get("generation", "Yanıt üretilemedi.")
            except Exception as e:
                answer = f"Bir hata oluştu: {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ingest import process_and_store_document

# Sayfa Yapılandırması
st.set_page_config(page_title="Hukuki RAG Asistanı", page_icon="⚖️", layout="wide")

st.title("⚖️ Kişiselleştirilmiş Hukuki Metin Soru-Cevap Asistanı")
st.write("Yasal metinlerinizi veya sözleşmelerinizi yükleyin, yapay zeka destekli asistanımız anında yanıtlasın.")

# --- SIDEBAR (YAN PANEL) - DOSYA YÜKLEME ---
with st.sidebar:
    st.header("📂 Belge Yönetimi")
    uploaded_file = st.file_uploader("Bir Kanun veya Sözleşme Yükleyin", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        # Geçici olarak diske kaydetme
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if st.button("Belgeyi İndeksle ve Hafızaya Al"):
            with st.spinner("Metinler parçalanıyor ve vektör veritabanına işleniyor..."):
                try:
                    chunk_count = process_and_store_document(file_path)
                    st.success(f"Başarılı! Toplam {chunk_count} parça hafızaya alındı.")
                except Exception as e:
                    st.error(f"Bir hata oluştu: {e}")

    st.markdown("---")
    st.markdown("**Not:** Sistem yerel kaynaklar (Ollama / Llama 3) üzerinden çalışmaktadır.")

# --- ANA EKRAN - SOHBET ALANI ---
@st.cache_resource
def load_rag_components():
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        return retriever
    return None

retriever = load_rag_components()

# LLM ve Zincir Yapılandırması
llm = ChatOllama(
    model="llama3",
    temperature=0,
    num_gpu=0  # CPU modu
)

template = """Sen uzman bir Türk hukuku asistanısın. 
Aşağıda verilen bağlam (kanun metinleri) bilgilerini dikkatlice oku ve kullanıcının sorusunu SADECE bu bağlama dayanarak yanıtla.
Eğer verilen metinde sorunun cevabı yoksa, kesinlikle dışarıdan bilgi uydurma ve "Verilen metitte bu bilgi bulunmamaktadır." de.
Mümkünse cevabının sonunda hangi maddeye dayandığını belirt.

Bağlam:
{context}

Soru: {question}

Cevap:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Sohbet Geçmişi (Streamlit State)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan Soru Alma
if user_query := st.chat_input("Hukuki metinle ilgili sorunuzu sorun..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if retriever is None:
            response = "⚠️ Lütfen önce sol panelden bir belge yükleyip indeksleyin!"
            st.markdown(response)
        else:
            with st.spinner("Yasal metinler taranıyor ve yanıt üretiliyor..."):
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                response = rag_chain.invoke(user_query)
                st.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})
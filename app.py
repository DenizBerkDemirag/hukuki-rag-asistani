import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage

# LangGraph ajanını içe aktarma
from graph_agent import legal_agent_app

st.set_page_config(page_title="Hukuki RAG Asistanı", page_icon="⚖️", layout="wide")

st.title("⚖️ Hukuki Metin Soru-Cevap ve Araştırma Asistanı")
st.caption("LangGraph Destekli Kendi Kendini Denetleyen Hukuk Ajanı")

# Oturum durumlarını (Session State) başlatma
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- YAN PANEL: PDF YÜKLEME VE İNDEKSLER ---
with st.sidebar:
    st.header("📄 Doküman Yönetimi")
    uploaded_file = st.file_uploader("Bir Hukuki PDF Yükleyin", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Belgeyi İndeksle"):
            with st.spinner("PDF işleniyor ve vektör veritabanına kaydediliyor..."):
                os.makedirs("data", exist_ok=True)
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # PDF Yükleme ve Parçalama
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = text_splitter.split_documents(docs)

                # ChromaDB'ye ekleme
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                st.success(f"Başarılı! {len(chunks)} parça ChromaDB'ye eklendi.")

    if st.button("Sohbet Geçmişini Temizle"):
        st.session_state.messages = []
        st.rerun()

# --- ANA EKRAN: SOHBET GEÇMİŞİ ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- KULLANICI GİRDİSİ (HİÇBİR BLOK İÇİNDE OLMAMALIDIR) ---
user_question = st.chat_input("Hukuki metinle ilgili bir soru sorun...")

if user_question:
    # 1. Kullanıcı mesajını ekrana ve geçmişe ekle
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Asistan cevabını oluştur
    with st.chat_message("assistant"):
        status_box = st.status("🔍 Ajan çalışıyor...", expanded=True)

        # Mesaj geçmişini LangChain formatına çevirme
        history_messages = []
        for msg in st.session_state.messages[:-1]:  # Son kullanıcı mesajı hariç önceki konuşmalar
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_messages.append(AIMessage(content=msg["content"]))

        inputs = {
            "question": user_question,
            "chat_history": history_messages
        }

        # LangGraph Memory için tekil thread ID
        config = {"configurable": {"thread_id": "session_single_user"}}
        final_answer = ""

        try:
            for output in legal_agent_app.stream(inputs, config=config):
                for node_name, node_state in output.items():
                    if node_name == "general_chat":
                        status_box.write("💬 **Sohbet:** Genel sohbet yanıtı oluşturuluyor...")
                        final_answer = node_state.get("generation", "")
                    elif node_name == "retrieve":
                        doc_count = len(node_state.get("documents", []))
                        status_box.write(f"📥 **Vektör Veritabanı:** {doc_count} ilgili metin parçası çekildi.")
                    elif node_name == "grade_documents":
                        valid_count = len(node_state.get("documents", []))
                        status_box.write(f"⚖️ **Hukuki Denetim:** {valid_count} parça soruyla alakalı onaylandı.")
                    elif node_name == "transform_query":
                        new_q = node_state.get("question", "")
                        status_box.write(f"🔄 **Sorgu İyileştirme:** Soru yenilendi: *'{new_q}'*")
                    elif node_name == "generate":
                        status_box.write("✍️ **Sentez:** Hukuki yanıt oluşturuldu.")
                        final_answer = node_state.get("generation", "")

            status_box.update(label="✅ Tamamlandı", state="complete", expanded=False)
            st.markdown(final_answer)

        except Exception as e:
            status_box.update(label="❌ Hata oluştu", state="error", expanded=True)
            st.error(f"Ajan çalışırken bir hata meydana geldi: {e}")
            final_answer = "Bir hata oluştu, lütfen konsolu kontrol edin."

    # 3. Asistan cevabını geçmişe kaydet
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
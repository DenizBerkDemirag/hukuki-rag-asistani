from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


print("RAG pipeline başlatılıyor...", flush=True)

# 1. Kaydedilmiş ChromaDB ve Embedding Modelini Yükleme
persist_directory = "./chroma_db"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# En yakın 2 parçayı getirecek retriever tanımlıyoruz
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 2. Yerelde Koşan Ollama Modeli (Llama 3)
llm = ChatOllama(
    model="llama3",
    temperature=0,   # Hukuki yanıtlarda uydurmayı (halüsinasyon) önlemek için deterministik tutuyoruz
    num_gpu=0  # GPU/CUDA çökmelerini engellemek için CPU üzerinden çalıştırıyoruz
)

# 3. Hukuki Prompt Şablonu
template = """Sen uzman bir Türk hukuku asistanısın. 
Aşağıda verilen bağlam (kanun metinleri) bilgilerini dikkatlice oku ve kullanıcının sorusunu SADECE bu bağlama dayanarak yanıtla.
Eğer verilen metinde sorunun cevabı yoksa, kesinlikle dışarıdan bilgi uydurma ve "Verilen metinde bu bilgi bulunmamaktadır." de.
Mümkünse cevabının sonunda hangi kanun maddesine dayandığını belirt.

Bağlam:
{context}

Soru: {question}

Cevap:"""

prompt = ChatPromptTemplate.from_template(template)

# Alınan doküman parçalarını tek bir metin haline getiren yardımcı fonksiyon
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. RAG Zincirini (LCEL - LangChain Expression Language) Oluşturma
rag_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Test Sorgusu
if __name__ == "__main__":
    soru="İş sözleşmesi fesinde bildirim süreleri neye göre belirlenir?"
    print(f"\nSoru: {soru}\n")
    print("Cevap alınıyor...", flush=True)

    cevap = rag_chain.invoke(soru)
    print("\n---Cevap---\n")
    print(cevap)
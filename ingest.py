import sys
print("Script başladı, kütüphaneler yükleniyor...", flush=True)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Kütüphaneler başarıyla yüklendi!", flush=True)

# 1. Dokümanı Yükleme
dosya_yolu = "data/ornek_kanun.txt"
print("Doküman yükleniyor...")
loader = TextLoader(dosya_yolu, encoding="utf-8")
documents = loader.load()

# 2. Metni Parçalara Bölme (Chunking)
print("Metin parçalanıyor...")
text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=300,  # Her parçanın yaklaşık karakter uzunluğu
                            chunk_overlap=50,# Parçalar arasındaki örtüşme miktarı (bağlam kopmasın diye)
                            separators=["\n\n", "\n", " ", ""],)
chunks = text_splitter.split_documents(documents)
print(f"Toplam {len(chunks)} adet parça (chunk) oluşturuldu.")

# 3. Embedding Modeli Seçimi (Hafif ve Türkçe için uyumlu açık kaynaklı model)
print("Embedding modeli seçiliyor...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. ChromaDB Vektör Veritabanına Kaydetme
persist_directory = "./chroma_db"
print("ChromaDB vektör veritabanına kaydediliyor...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("Vektör veritabanı başarıyla oluşturuldu ve kaydedildi.")
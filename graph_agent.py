import re
from typing import List, Sequence
from typing_extensions import TypedDict
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# 1. State Tanımı
class GraphState(TypedDict):
    question: str
    standalone_question: str
    generation: str
    documents: List[Document]
    retry_count: int
    chat_history: Sequence[BaseMessage]

# 2. Bileşenler
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model="llama3", temperature=0)

# --- DÜĞÜMLER (NODES) ---

def contextualize_question_node(state: GraphState):
    """
    Takip sorularını konuşma geçmişiyle birleştirip tek başına bağımsız,
    eksiksiz bir hukuki araştırma sorusuna dönüştürür.
    """
    question = state["question"]
    history = state.get("chat_history", [])
    
    if not history:
        return {"standalone_question": question}
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Aşağıdaki konuşma geçmişini ve kullanıcının son sorusunu incele.
Eğer son soru önceki konuşmaya atıfta bulunuyorsa (örneğin 'bunun cezası nedir', 'istisnaları nelerdir' gibi), 
soruyu tek başına anlaşılır ve arama yapmaya uygun tam bir hukuki soruya dönüştür.
Eğer soru zaten bağımsızsa olduğu gibi bırak. YALNIZCA dönüştürülmüş tek cümlelik soruyu yaz, açıklama yapma."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    standalone = chain.invoke({"chat_history": history, "question": question}).strip()
    return {"standalone_question": standalone if standalone else question}

def general_chat_node(state: GraphState):
    """Genel sohbet yanıtı üretir."""
    question = state["question"]
    history = state.get("chat_history", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sen yardımcı, kibar bir Hukuki Araştırma Asistanısın. Türkçe konuş. Kullanıcının sorusuna doğrudan ve nazikçe yanıt ver."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    generation = (prompt | llm | StrOutputParser()).invoke({"question": question, "chat_history": history})
    return {"generation": generation}

def retrieve_node(state: GraphState):
    """Bağımsızlaştırılmış soru üzerinden vektör veritabanından belge çeker."""
    query = state.get("standalone_question") or state["question"]
    docs = retriever.invoke(query)
    retry_count = state.get("retry_count", 0)
    return {"documents": docs, "retry_count": retry_count}

def grade_documents_node(state: GraphState):
    """
    Tüm dokümanları tek bir LLM çağrısıyla toplu olarak denetler (Batch Grading).
    Böylece 4 ayrı çağrı yerine tek çağrı yapılarak hız ve kararlılık katbekat artar.
    """
    query = state.get("standalone_question") or state["question"]
    documents = state.get("documents", [])
    
    if not documents:
        return {"documents": []}
        
    doc_snippets = []
    for i, doc in enumerate(documents, 1):
        content_preview = doc.page_content[:300].replace("\n", " ")
        doc_snippets.append(f"[{i}] {content_preview}")
        
    docs_text = "\n".join(doc_snippets)
    
    prompt = ChatPromptTemplate.from_template(
        """Verilen soru ile aşağıdaki doküman özetlerini karşılaştır.
Soruyla anlamsal veya hukuki bağı olan dokümanların sadece numaralarını virgülle yaz (Örn: 1, 2).
Hiçbiri ilgili değilse 'yok' yaz.

Soru: {question}

Dokümanlar:
{docs_text}

İlgili Numaralar:"""
    )
    
    decision = (prompt | llm | StrOutputParser()).invoke({
        "question": query,
        "docs_text": docs_text
    }).strip().lower()
    
    filtered_docs = []
    if "yok" not in decision:
        # LLM'in döndürdüğü numaraları ayıkla
        numbers = re.findall(r"\b[1-4]\b", decision)
        for num_str in set(numbers):
            idx = int(num_str) - 1
            if 0 <= idx < len(documents):
                filtered_docs.append(documents[idx])
                
    # Fallback: Eğer LLM format dışı yanıt verdiyse veya filtre boş kaldıysa ilk 2 dokümanı koru
    if not filtered_docs and len(documents) > 0 and "yok" not in decision:
        filtered_docs = documents[:2]
        
    return {"documents": filtered_docs}

def transform_query_node(state: GraphState):
    """Arama ifadesini hukuki terminolojiye uygun hale getirir."""
    query = state.get("standalone_question") or state["question"]
    retry_count = state.get("retry_count", 0) + 1
    
    prompt = ChatPromptTemplate.from_template(
        """Aşağıdaki soruyu kanun maddelerinde arama yapmaya uygun, anahtar hukuki kavramları içeren tek cümlelik bir arama ifadesine dönüştür. Yalnızca Türkçe yaz.
Soru: {question}
Hukuki Arama İfadesi:"""
    )
    better_query = (prompt | llm | StrOutputParser()).invoke({"question": query}).strip()
    return {"standalone_question": better_query, "retry_count": retry_count}

def generate_node(state: GraphState):
    """Hukuki yanıt üretir ve kaynak atıflarını ekler."""
    question = state["question"]
    documents = state.get("documents", [])
    history = state.get("chat_history", [])
    
    # Güvenlik önlemi: Hiç belge yoksa modeli boş bağlamla zorlamadan net mesaj dön
    if not documents:
        empty_msg = "Yüklenen hukuki belgelerde bu konuya ilişkin ilgili bir hüküm veya bilgi bulunamamaktadır. Lütfen sorunuzu farklı kavramlarla sormayı deneyin veya ilgili belgenin yüklendiğinden emin olun."
        return {"generation": empty_msg, "documents": []}
        
    formatted_docs = []
    sources_summary = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Belge")
        page = doc.metadata.get("page", None)
        page_info = f" (Sayfa {page + 1})" if page is not None else ""
        formatted_docs.append(f"--- [Belge {i} | {source}{page_info}] ---\n{doc.page_content}")
        sources_summary.append(f"- {source}{page_info}")
        
    context = "\n\n".join(formatted_docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Sen Türk hukuku alanında uzman bir araştırma asistanısın. 
Verilen bağlam bilgilerine dayanarak soruyu SADECE TÜRKÇE olarak, açık ve gerekçeli bir dille yanıtla.
Mümkünse ilgili fıkra, madde ve sayfa numaralarına atıfta bulun.
Metinde net bir karşılık yoksa kesinlikle varsayımda bulunma; 'Verilen belgelerde bu konuya ilişkin bilgi bulunmamaktadır.' de.

Bağlam:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    generation = (prompt | llm | StrOutputParser()).invoke({
        "context": context, 
        "question": question,
        "chat_history": history
    })
    
    # Kullanılan kaynakları cevabın sonuna ekleme
    if sources_summary and "Verilen belgelerde bu konuya ilişkin bilgi bulunmamaktadır." not in generation:
        unique_sources = list(dict.fromkeys(sources_summary))
        generation += "\n\n📌 **Başvurulan Kaynaklar:**\n" + "\n".join(unique_sources)
        
    return {"generation": generation, "documents": documents}

# --- KARAR MEKANİZMALARI ---

def route_question(state: GraphState):
    query = state.get("standalone_question") or state["question"]
    prompt = ChatPromptTemplate.from_template(
        """Kullanıcının iletisini sınıflandır. 
Eğer bu bir selamlama, hal hatır sorma, genel sohbet ise 'chat' yaz.
Eğer kanun, mevzuat, avukatlık veya hukuki bir soru ise 'legal' yaz.
Sadece 'chat' veya 'legal' çıktısı ver.

İleti: {query}
Sınıflandırma:"""
    )
    decision = (prompt | llm | StrOutputParser()).invoke({"query": query}).strip().lower()
    if "chat" in decision:
        return "general_chat"
    return "retrieve"

def decide_to_generate(state: GraphState):
    filtered_docs = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    if not filtered_docs:
        if retry_count >= 1:
            return "generate"
        return "transform_query"
    return "generate"

# --- GRAFİK KURULUMU ---

workflow = StateGraph(GraphState)

workflow.add_node("contextualize", contextualize_question_node)
workflow.add_node("general_chat", general_chat_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("generate", generate_node)

# İlk adım her zaman soruyu bağlama oturtmaktır
workflow.set_entry_point("contextualize")

# Bağlamlaştırmadan sonra yönlendirme kararı verilir
workflow.add_conditional_edges(
    "contextualize",
    route_question,
    {"general_chat": "general_chat", "retrieve": "retrieve"}
)

workflow.add_edge("general_chat", END)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"transform_query": "transform_query", "generate": "generate"}
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

memory = MemorySaver()
legal_agent_app = workflow.compile(checkpointer=memory)
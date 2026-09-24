from typing import List, Sequence
from typing_extensions import TypedDict
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# 1. State Tanımı: Artık mesaj geçmişini (chat_history) de tutuyor
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retry_count: int
    chat_history: Sequence[BaseMessage]

# 2. Bileşenler
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOllama(model="llama3", temperature=0)

# --- DÜĞÜMLER (NODES) ---

def general_chat_node(state: GraphState):
    """Sohbet geçmişini de dikkate alarak yanıt verir."""
    question = state["question"]
    history = state.get("chat_history", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sen yardımcı, kibar bir Hukuki Araştırma Asistanısın. Türkçe konuş. Önceki konuşmaları hatırla ve kullanıcının adıyla veya bağlamıyla hitap et."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"question": question, "chat_history": history})
    return {"generation": generation}

def retrieve_node(state: GraphState):
    question = state["question"]
    docs = retriever.invoke(question)
    doc_texts = [d.page_content for d in docs]
    retry_count = state.get("retry_count", 0)
    return {"documents": doc_texts, "question": question, "retry_count": retry_count}

def grade_documents_node(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    
    prompt = ChatPromptTemplate.from_template(
        """Sen bir denetleyicisin. Doküman verilen soruyla anlamsal olarak ilgili mi?
Sadece 'evet' ya da 'hayir' yaz.

Doküman: {document}
Soru: {question}
Cevap (evet/hayir):"""
    )
    grader_chain = prompt | llm | StrOutputParser()
    
    filtered_docs = []
    for doc in documents:
        score = grader_chain.invoke({"question": question, "document": doc})
        if "evet" in score.lower():
            filtered_docs.append(doc)
            
    return {"documents": filtered_docs, "question": question}

def transform_query_node(state: GraphState):
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1
    
    prompt = ChatPromptTemplate.from_template(
        """Kullanıcının sorusunu kanun maddelerinde arama yapmaya uygun, tek cümlelik bir hukuki arama ifadesine dönüştür. Yalnızca Türkçe yaz.
Soru: {question}
Yenilenmiş Soru:"""
    )
    better_question = (prompt | llm | StrOutputParser()).invoke({"question": question})
    return {"question": better_question.strip(), "retry_count": retry_count}

def generate_node(state: GraphState):
    """Hukuki yanıt üretirken de geçmişi göz önüne alır."""
    question = state["question"]
    documents = state["documents"]
    history = state.get("chat_history", [])
    context = "\n\n".join(documents)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Sen Türk hukuku alanında uzman bir asistansın. 
Verilen bağlam bilgilerine dayanarak soruyu SADECE TÜRKÇE olarak yanıtla. 
Metinde net bir karşılık yoksa 'Verilen belgelerde bu konuya ilişkin bilgi bulunmamaktadır.' de.

Bağlam:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    generator_chain = prompt | llm | StrOutputParser()
    generation = generator_chain.invoke({
        "context": context, 
        "question": question,
        "chat_history": history
    })
    return {"generation": generation}

# --- KARAR MEKANİZMALARI ---

def route_question(state: GraphState):
    question = state["question"]
    prompt = ChatPromptTemplate.from_template(
        """Kullanıcının iletisini sınıflandır. 
Eğer bu bir selamlama, hal hatır sorma, isim sorma, genel sohbet veya önceki konuşmaya atıf ise 'chat' yaz.
Eğer kanun, mevzuat, avukatlık veya hukuki bir konu hakkında soru ise 'legal' yaz.
Sadece 'chat' veya 'legal' çıktısı ver.

İleti: {question}
Sınıflandırma:"""
    )
    router_chain = prompt | llm | StrOutputParser()
    decision = router_chain.invoke({"question": question}).strip().lower()
    
    if "chat" in decision:
        return "general_chat"
    return "retrieve"

def decide_to_generate(state: GraphState):
    filtered_docs = state["documents"]
    retry_count = state.get("retry_count", 0)
    if not filtered_docs:
        if retry_count >= 1:
            return "generate"
        return "transform_query"
    return "generate"

# --- GRAFİK KURULUMU ---

workflow = StateGraph(GraphState)

workflow.add_node("general_chat", general_chat_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("generate", generate_node)

workflow.set_conditional_entry_point(
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

# Checkpointer ekleyerek hafızayı derliyoruz
memory = MemorySaver()
legal_agent_app = workflow.compile(checkpointer=memory)
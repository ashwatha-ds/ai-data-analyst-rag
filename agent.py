from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()

class ChatAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.retriever = None

    def initialize_retriever(self, rag_handler):
        self.retriever = rag_handler.get_retriever()

    def get_response(self, question: str):
        if not self.retriever:
            return "Please generate a report first before chatting."

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Data Analyst. Use the provided report context to answer questions accurately.
            Be professional, clear, and provide insights when possible."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        chain = prompt | self.llm

        history = ChatMessageHistory()
        chain_with_memory = RunnableWithMessageHistory(
            chain,
            lambda session_id: history,
            input_messages_key="input",
            history_messages_key="history"
        )

        context_docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in context_docs])

        response = chain_with_memory.invoke(
            {"input": f"Context:\n{context}\n\nQuestion: {question}"},
            config={"configurable": {"session_id": "default"}}
        )

        return response.content

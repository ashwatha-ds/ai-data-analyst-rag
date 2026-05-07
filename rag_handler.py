import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class RAGHandler:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.persist_directory = "chroma_db"
        self.vectorstore = None

    def index_report(self, markdown_path: str, filename: str):
        if not os.path.exists(markdown_path):
            return False
        loader = TextLoader(markdown_path, encoding="utf-8")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)

        for split in splits:
            split.metadata["source"] = filename

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(splits)
        return True

    def get_retriever(self):
        if self.vectorstore is None and os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self.vectorstore.as_retriever(search_kwargs={"k": 6}) if self.vectorstore else None

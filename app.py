import streamlit as st
import pandas as pd
import os
import sys

# ChromaDB SQLite fix — only needed on Linux (Streamlit Cloud)
if sys.platform.startswith("linux"):
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass

from report_generator import ReportGenerator
from rag_handler import RAGHandler
from agent import ChatAgent

st.set_page_config(page_title="AutoEDA Agent", layout="wide")
st.title("AutoEDA Agent")
st.markdown("**AI-Powered Data Analysis Report + Intelligent Chat**")

# Initialize ALL session state at the top
if "report_generator" not in st.session_state:
    st.session_state.report_generator = ReportGenerator()
if "rag_handler" not in st.session_state:
    st.session_state.rag_handler = RAGHandler()
if "chat_agent" not in st.session_state:
    st.session_state.chat_agent = ChatAgent()
if "current_md_path" not in st.session_state:
    st.session_state.current_md_path = None
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "report_ready" not in st.session_state:
    st.session_state.report_ready = False

# Sidebar
with st.sidebar:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file and st.button("Generate Professional Report", type="primary"):
        with st.spinner("Analyzing data and generating report... This may take 20-40 seconds"):
            try:
                if uploaded_file.name.endswith('.csv'):
                    try:
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                else:
                    df = pd.read_excel(uploaded_file)

                filename = uploaded_file.name.replace(".csv", "").replace(".xlsx", "").replace(".xls", "")

                markdown_content, md_path, pdf_path = st.session_state.report_generator.generate_full_report(df, filename)

                st.session_state.current_md_path = md_path
                st.session_state.current_filename = filename
                st.session_state.report_ready = True

                # Index for RAG
                st.session_state.rag_handler.index_report(md_path, filename)
                st.session_state.chat_agent.initialize_retriever(st.session_state.rag_handler)

                st.success("Report Generated Successfully!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Main Tabs
tab1, tab2 = st.tabs(["Generated Report", "Chat with Analysis Agent"])

with tab1:
    if st.session_state.current_md_path and os.path.exists(st.session_state.current_md_path):
        st.subheader(f"Report: {st.session_state.current_filename}")

        with open(st.session_state.current_md_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())

        col1, col2 = st.columns(2)
        with col1:
            with open(st.session_state.current_md_path, "r", encoding="utf-8") as f:
                st.download_button("Download Markdown", f.read(), f"{st.session_state.current_filename}_report.md", "text/markdown")

        with col2:
            pdf_path = f"reports/{st.session_state.current_filename}_report.pdf"
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF Report", f.read(), f"{st.session_state.current_filename}_report.pdf", "application/pdf")
    else:
        st.info("Upload a CSV or Excel file from the sidebar and generate the report")

with tab2:
    st.subheader("Ask Questions about the Dataset")

    if not st.session_state.report_ready:
        st.warning("Please generate a report first to enable chat")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask any question about the data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chat_agent.get_response(prompt)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

st.caption("AutoEDA Agent | Groq + LangChain + RAG")

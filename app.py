import os
import textwrap
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



# ----------------------------
# Config
# ----------------------------
CSV_FILE_PATH = "explanations_data.csv"
CHUNK_SIZE = 3000
TOP_K = 4
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.3


def wrap_text_preserve_newlines(text, width=110):
    lines = text.split("\n")
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    return "\n".join(wrapped_lines)


@st.cache_resource(show_spinner=True)
def build_qa_pipeline():
    """Build once and reuse (vector DB + retriever + LLM chain)."""
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE_PATH}")

    loader = CSVLoader(CSV_FILE_PATH, encoding="utf-8")
    data = loader.load()

    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    docs = splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=google_api_key
    )

    db = FAISS.from_documents(docs, embeddings)

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        google_api_key=google_api_key
    )

    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_template(
            """Use the context below to answer the question.
    If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """
        )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain,retriever


def main():
    st.set_page_config(page_title="Agro Q&A App", page_icon="🌿", layout="centered")

    st.title("🌿 Agro Q&A App")
    st.caption("Ask questions from your CSV knowledge base (RAG with FAISS + Gemini).")

    # Build pipeline once
    qa,retriever = build_qa_pipeline()

    # Input
    question = st.text_input("Enter your question", placeholder="e.g., What is the cropping system?")

    col1, col2 = st.columns([1, 1])
    with col1:
        ask = st.button("Get Answer", use_container_width=True)
    with col2:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.session_state["answer"] = ""
        st.session_state["question"] = ""
        st.rerun()

    if ask:
        q = question.strip()
        if not q:
            st.warning("Please type a question.")
        else:
            with st.spinner("Thinking..."):
                result = qa.invoke(q)
                formatted = wrap_text_preserve_newlines(result)

            st.session_state["question"] = q
            st.session_state["answer"] = formatted

    # Output
    if "question" in st.session_state and st.session_state["question"]:
        st.subheader("Your Question")
        st.write(st.session_state["question"])

    if "answer" in st.session_state and st.session_state["answer"] != "":
        st.subheader("Answer")
        st.code(st.session_state["answer"], language="text")


if __name__ == "__main__":
    main()

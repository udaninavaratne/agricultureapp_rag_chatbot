import os
import textwrap
import yaml
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnableLambda
from sentence_transformers import CrossEncoder

#Configurating prameters
def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)

config = load_config()  

CSV_FILE_PATH = config["data"]["csv_file_path"]
FAISS_DIR = config["data"]["faiss_dir"]

CHUNK_SIZE = config["chunking"]["chunk_size"]
CHUNK_OVERLAP = config["chunking"]["chunk_overlap"]

TOP_K = config["retrieval"]["top_k"]
FETCH_K = config["retrieval"]["fetch_k"]

MODEL_NAME = config["llm"]["model_name"]
EMBEDDING_MODEL = config["llm"]["embedding_model"]
TEMPERATURE = config["llm"]["temperature"]

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split("\n")
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    return "\n".join(wrapped_lines)

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def _save_uploaded_files_to_temp(uploaded_files):
    """Save Streamlit uploaded files to temp paths and return list of file paths."""
    tmp_dir = tempfile.mkdtemp()
    paths = []
    for uf in uploaded_files:
        path = os.path.join(tmp_dir, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        paths.append(path)
    return paths

def _load_docs_from_paths(paths):
    docs = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()

        if ext == ".csv":
            docs.extend(CSVLoader(p).load())
        elif ext == ".pdf":
            docs.extend(PyPDFLoader(p).load())
        elif ext in [".txt", ".md"]:
            docs.extend(TextLoader(p, encoding="utf-8").load())
        else:
            # skip unsupported for now
            continue
    return docs

@st.cache_resource(show_spinner=True)
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource(show_spinner=True)
def build_qa_pipeline(uploaded_files=None,_key=None):
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    if not uploaded_files and not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE_PATH}")

    # ----------------------------
    # 1) Load docs
    # ----------------------------
    if uploaded_files:
        paths = _save_uploaded_files_to_temp(uploaded_files)
        docs = _load_docs_from_paths(paths)
    else:
        docs = CSVLoader(CSV_FILE_PATH).load()

    # ----------------------------
    # 2) Chunking
    # ----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    print("Total chunks:", len(chunks))
    if chunks:
        print(chunks[0].page_content[:500])

    # ----------------------------
    # 3) Embeddings
    # ----------------------------
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    # ----------------------------
    # 4) Vector DB (FAISS)
    #   - If uploads exist: build from chunks in-memory (don’t reuse old FAISS_DIR)
    #   - If no uploads: use your saved FAISS_DIR logic
    # ----------------------------
    if uploaded_files:
        db = FAISS.from_documents(chunks, embeddings)
    else:
        if os.path.exists(FAISS_DIR):
            db = FAISS.load_local(
                FAISS_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(FAISS_DIR)

    # ----------------------------
    # 5) Retriever + reranker
    # ----------------------------
    reranker = get_reranker()
    base_retriever = db.as_retriever(search_kwargs={"k": FETCH_K})

    def expand_queries(question: str) -> list[str]:
        qe = config.get("query_expansion", {})
        enabled = bool(qe.get("enabled", True))
        if not enabled:
            return [question]

        num_queries = int(qe.get("num_queries", 4))
        exp_model = qe.get("model_name", "gpt-4.1-mini")
        exp_temp = float(qe.get("temperature", 0.2))

        expander_llm = ChatOpenAI(
            model=exp_model,
            temperature=exp_temp,
            openai_api_key=OPENAI_API_KEY
        )

        expand_prompt = ChatPromptTemplate.from_template(
            """Generate {n} alternative search queries for the question below.

Rules:
- Each query must be short (<= 12 words)
- Use different wording, synonyms, and more specific versions
- No numbering, no bullets, one query per line
- Do not include extra commentary

Question: {q}
"""
        )

        msg = expander_llm.invoke(expand_prompt.format_messages(n=num_queries, q=question))
        raw = msg.content.strip()

        candidates = [line.strip() for line in raw.split("\n") if line.strip()]

        all_queries = [question] + candidates
        seen = set()
        unique = []
        for qq in all_queries:
            key = qq.lower()
            if key not in seen:
                seen.add(key)
                unique.append(qq)

        return unique[: num_queries + 1]

    def rerank_docs(query, docs):
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:TOP_K]]

    def retrieve_and_rerank(query: str):
        expanded_queries = expand_queries(query)

        all_docs = []
        for q in expanded_queries:
            all_docs.extend(base_retriever.invoke(q))

        unique = {}
        for d in all_docs:
            meta = tuple(sorted(d.metadata.items())) if d.metadata else ()
            key = (d.page_content, meta)
            unique[key] = d

        merged_docs = list(unique.values())
        return rerank_docs(query, merged_docs)

    # ----------------------------
    # 6) LLM + prompt
    # ----------------------------
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )

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

    def get_context(question):
        docs = retrieve_and_rerank(question)
        return format_docs(docs)

    context_runnable = RunnableLambda(get_context)

    chain = (
        {"context": context_runnable, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, base_retriever

def _corpus_key(uploaded_files):
    # helps st.cache_resource know when uploads changed
    if not uploaded_files:
        return None
    return tuple((f.name, f.size) for f in uploaded_files)

def main():
    # Set layout
    st.set_page_config(page_title="Agro Q&A App", page_icon="🌿", layout="centered")

    st.title("🤖Q&A App")
    st.caption("RAG + FAISS + Cross-Encoder Re-ranking")

    # 1) Upload gate (REQUIRED)
    uploaded_files = st.file_uploader(
        "Upload documents to start (PDF, CSV, TXT, MD)",
        type=["pdf", "csv", "txt", "md"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("⬆️ Upload at least one document to start asking questions.")
        st.stop()

    # 2) Build pipeline from uploads
    qa, retriever = build_qa_pipeline(
        uploaded_files=uploaded_files,
        _key=_corpus_key(uploaded_files)   # only if you added _key in signature
    )
    # If you DID NOT add _key to build_qa_pipeline signature, use:
    # qa, retriever = build_qa_pipeline(uploaded_files=uploaded_files)

    # 3) Question input
    question = st.text_input("Enter your question", key="question_input")

    col1, col2 = st.columns([1, 1])
    with col1:
        ask = st.button("Get Answer", use_container_width=True)
    with col2:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.session_state["answer"] = ""
        st.session_state["question"] = ""
        st.session_state["question_input"] = ""
        st.rerun()

    if ask:
        q = (question or "").strip()
        if not q:
            st.warning("Please type a question.")
        else:
            with st.spinner("Thinking..."):
                result = qa.invoke(q)
                formatted = wrap_text_preserve_newlines(result)

            st.session_state["question"] = q
            st.session_state["answer"] = formatted

    if st.session_state.get("question"):
        st.subheader("Your Question")
        st.write(st.session_state["question"])

    if st.session_state.get("answer"):
        st.subheader("Answer")
        st.code(st.session_state["answer"], language="text")

if __name__ == "__main__":
    main()
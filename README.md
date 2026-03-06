Project NameRAG-based Q&A Application

DescriptionThis project is a Streamlit application that allows users to upload documents and ask questions about them. The system retrieves relevant information from the documents and generates answers using an LLM.

Main Features

*   Upload documents (PDF, CSV, TXT, MD)
    
*   Ask questions about the uploaded files
    
*   Retrieve relevant document sections
    
*   Generate answers using AI
    
*   Filter results by file
    

Technologies Used

*   Python
    
*   Streamlit
    
*   LangChain
    
*   FAISS
    
*   OpenAI
    
*   Sentence Transformers
    

How It Works

1.  User uploads documents.
    
2.  Documents are split into smaller chunks.
    
3.  The system retrieves relevant chunks using FAISS and BM25.
    
4.  A reranker selects the most relevant results.
    
5.  The LLM generates the final answer.
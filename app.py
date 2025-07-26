import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  # Or HuggingFaceEmbeddings for open-source
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # Compatible with Grok API
from langchain.prompts import PromptTemplate

# Initialize session state for persistence
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'llm' not in st.session_state:
    st.session_state.llm = OpenAI(
        api_key=os.getenv("GROK_API_KEY"),  # Your xAI key via env var
        base_url="https://api.x.ai/v1"      # xAI endpoint
    )

st.title("AI Document Analyzer")

# Step 1: Upload and Learn Spec Book
st.header("Upload Spec Book")
spec_file = st.file_uploader("Upload Spec Book (PDF or DOCX)", type=["pdf", "docx"])
if spec_file and st.button("Learn Spec Book"):
    with st.spinner("Indexing spec book..."):
        # Load document
        if spec_file.type == "application/pdf":
            loader = PyPDFLoader(spec_file)
        else:
            loader = Docx2txtLoader(spec_file)
        docs = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Embed and index
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Using HuggingFace instead of OpenAI
        st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
    st.success("Spec book learned!")

# Step 2: Upload Audit Document and Analyze
st.header("Upload Audit Document")
audit_file = st.file_uploader("Upload Audit (PDF or DOCX)", type=["pdf", "docx"])
if audit_file and st.session_state.vectorstore and st.button("Analyze Audit"):
    with st.spinner("Analyzing audit..."):
        # Load audit
        if audit_file.type == "application/pdf":
            loader = PyPDFLoader(audit_file)
        else:
            loader = Docx2txtLoader(audit_file)
        audit_docs = loader.load()
        
        # Dummy extraction of infractions (customize this to parse your audit format, e.g., look for "go-back" keywords)
        audit_text = " ".join([doc.page_content for doc in audit_docs])
        infractions = [line.strip() for line in audit_text.split("\n") if "go-back" in line.lower()]  # Simple keyword-based; improve with regex/NLP
        
        # Set up RAG chain
        prompt_template = """
        Analyze this infraction: {infraction}
        Cross-reference with the spec book context: {context}
        
        Output:
        - Valid: Yes/No
        - Repealable: Yes/No (with confidence 0-100%)
        - If repealable, list reasons from spec book.
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["infraction", "context"])
        qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        results = []
        for infraction in infractions:
            result = qa_chain.run({"infraction": infraction})  # LangChain handles context injection
            results.append(result)
    
    st.subheader("Analysis Results")
    for i, res in enumerate(results):
        st.write(f"Infraction {i+1}: {infractions[i]}")
        st.write(res)
        st.write("---") 
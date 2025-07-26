import streamlit as st
import os
import tempfile
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI  # Compatible with Grok API
from langchain.prompts import PromptTemplate

# Initialize session state for persistence
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'llm' not in st.session_state:
    st.session_state.llm = OpenAI(
        api_key=os.getenv("GROK_API_KEY"),  # Your xAI key via env var
        base_url="https://api.x.ai/v1",      # xAI endpoint
        model="grok-2-latest",
        temperature=0.3  # Lower temperature for more consistent analysis
    )

st.title("QA Audit Document Analyzer")

# Step 1: Upload and Learn Spec Book
st.header("Step 1: Upload Spec Book")
spec_file = st.file_uploader("Upload Spec Book (PDF or DOCX)", type=["pdf", "docx"])
if spec_file and st.button("Index Spec Book"):
    with st.spinner("Indexing spec book..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(spec_file.name)[1]) as tmp_file:
            tmp_file.write(spec_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load document
        if spec_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = Docx2txtLoader(tmp_file_path)
        docs = loader.load()
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        # Embed and index
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
    st.success(f"✅ Spec book indexed! ({len(splits)} chunks created)")

# Step 2: Upload Audit Document and Analyze
st.header("Step 2: Upload QA Audit Document")
audit_file = st.file_uploader("Upload QA Audit (PDF or DOCX)", type=["pdf", "docx"])

def extract_infractions(audit_docs):
    """Extract infractions from audit document starting from page 2"""
    infractions = []
    
    # Combine all pages into text
    full_text = ""
    for i, doc in enumerate(audit_docs):
        if i >= 1:  # Start from page 2 (0-indexed)
            full_text += f"\n--- Page {i+1} ---\n" + doc.page_content
    
    # Look for flagged items section
    flagged_section = re.search(r'flagged\s*items|infractions|deficiencies|non.?conformances', 
                                full_text, re.IGNORECASE)
    
    if flagged_section:
        # Extract text after flagged items header
        flagged_text = full_text[flagged_section.end():]
        
        # Pattern to match infractions with spec references
        # This pattern looks for numbered items or bullet points followed by spec references
        patterns = [
            r'(\d+\.?\s*[^\n]+?(?:spec|standard|reference|section)[^\n]+?page\s*\d+[^\n]*)',
            r'([•·-]\s*[^\n]+?(?:spec|standard|reference|section)[^\n]+?page\s*\d+[^\n]*)',
            r'((?:infraction|deficiency|issue|finding)[\s\S]{0,500}?(?:spec|standard|reference)[^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, flagged_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                infraction_data = {
                    'text': match.strip(),
                    'spec_ref': extract_spec_reference(match),
                    'page_num': extract_page_number(match),
                    'reasoning': extract_reasoning(match)
                }
                infractions.append(infraction_data)
    
    # If no flagged section found, try to extract any items that look like infractions
    if not infractions:
        # Fallback pattern for infractions
        fallback_pattern = r'(?:^|\n)(?:\d+\.?|[•·-])\s*([^\n]+(?:spec|standard|violation|deficiency|non.?conformance)[^\n]+)'
        matches = re.findall(fallback_pattern, full_text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            infraction_data = {
                'text': match.strip(),
                'spec_ref': extract_spec_reference(match),
                'page_num': extract_page_number(match),
                'reasoning': 'No specific reasoning provided in audit'
            }
            infractions.append(infraction_data)
    
    return infractions

def extract_spec_reference(text):
    """Extract specification reference from infraction text"""
    spec_patterns = [
        r'(?:spec|standard|section)\s*([A-Z0-9.-]+)',
        r'(?:reference|ref\.?)\s*([A-Z0-9.-]+)',
        r'([A-Z]{2,}\s*\d+(?:\.\d+)*)'
    ]
    
    for pattern in spec_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return "No spec reference found"

def extract_page_number(text):
    """Extract page number from infraction text"""
    page_pattern = r'page\s*(\d+)'
    match = re.search(page_pattern, text, re.IGNORECASE)
    return match.group(1) if match else "N/A"

def extract_reasoning(text):
    """Extract reasoning from infraction text"""
    reasoning_patterns = [
        r'(?:because|due to|reason|since)\s*([^.]+)',
        r'(?:found|observed|noted)\s*([^.]+)'
    ]
    
    for pattern in reasoning_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "See audit document for details"

if audit_file and st.session_state.vectorstore and st.button("Analyze Audit"):
    with st.spinner("Analyzing audit document..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audit_file.name)[1]) as tmp_file:
            tmp_file.write(audit_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load audit
        if audit_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = Docx2txtLoader(tmp_file_path)
        audit_docs = loader.load()
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Extract infractions
        infractions = extract_infractions(audit_docs)
        
        if not infractions:
            st.warning("No infractions found in the audit document. Please check if the document contains flagged items starting from page 2.")
        else:
            st.info(f"Found {len(infractions)} potential infractions to analyze")
        
        # Set up RAG chain with detailed prompt
        prompt_template = """You are an expert QA auditor analyzing infractions against specification standards.

Infraction Details:
- Infraction: {infraction}
- Spec Reference: {spec_ref}
- Page Number: {page_num}
- Auditor's Reasoning: {reasoning}

Relevant Spec Book Context:
{context}

Please provide a detailed analysis with the following structure:

1. **VALIDITY ASSESSMENT**: Is this a true infraction? (YES/NO)

2. **CONFIDENCE RATING**: Rate your confidence from 0-100%

3. **CAN BE OVERTURNED**: YES/NO

4. **DETAILED ANALYSIS**:
   - Cross-reference the specific spec section mentioned
   - Identify any ambiguities or interpretations
   - Check if the auditor's reasoning aligns with spec requirements

5. **IF CAN BE OVERTURNED - REASONING**:
   - Specific spec sections that support overturning
   - Alternative interpretations of the spec
   - Precedents or exceptions that apply
   - Recommended argument structure

6. **IF CANNOT BE OVERTURNED - REASONING**:
   - Clear spec violations identified
   - Why the infraction is valid per specifications
   - Corrective actions recommended

Be thorough and cite specific sections from the spec book in your analysis."""

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["infraction", "spec_ref", "page_num", "reasoning", "context"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # Analyze each infraction
        results = []
        progress_bar = st.progress(0)
        
        for i, infraction_data in enumerate(infractions):
            # Prepare the query with all infraction details
            query = f"""Infraction: {infraction_data['text']}
Spec Reference: {infraction_data['spec_ref']}
Page Number: {infraction_data['page_num']}
Reasoning: {infraction_data['reasoning']}"""
            
            result = qa_chain.invoke({
                "query": query,
                "infraction": infraction_data['text'],
                "spec_ref": infraction_data['spec_ref'],
                "page_num": infraction_data['page_num'],
                "reasoning": infraction_data['reasoning']
            })
            
            results.append({
                'infraction': infraction_data,
                'analysis': result['result'] if isinstance(result, dict) else str(result)
            })
            
            progress_bar.progress((i + 1) / len(infractions))
    
    # Display results
    st.header("📊 Analysis Results")
    
    # Summary statistics
    overturn_count = sum(1 for r in results if "CAN BE OVERTURNED: YES" in r['analysis'].upper())
    valid_count = sum(1 for r in results if "VALIDITY ASSESSMENT: YES" in r['analysis'].upper())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Infractions", len(results))
    with col2:
        st.metric("Valid Infractions", valid_count)
    with col3:
        st.metric("Can Be Overturned", overturn_count)
    
    # Detailed results
    for i, result in enumerate(results):
        with st.expander(f"Infraction {i+1}: {result['infraction']['spec_ref']} (Page {result['infraction']['page_num']})"):
            st.subheader("Infraction Details")
            st.write(f"**Text:** {result['infraction']['text']}")
            st.write(f"**Spec Reference:** {result['infraction']['spec_ref']}")
            st.write(f"**Page Number:** {result['infraction']['page_num']}")
            st.write(f"**Auditor's Reasoning:** {result['infraction']['reasoning']}")
            
            st.subheader("AI Analysis")
            st.write(result['analysis'])
            
            # Highlight key findings
            if "CAN BE OVERTURNED: YES" in result['analysis'].upper():
                st.success("✅ This infraction can potentially be overturned")
            else:
                st.error("❌ This infraction appears to be valid")
    
    # Export results option
    if st.button("Export Analysis to Text File"):
        export_text = "QA AUDIT ANALYSIS REPORT\n" + "="*50 + "\n\n"
        for i, result in enumerate(results):
            export_text += f"\nINFRACTION {i+1}\n" + "-"*30 + "\n"
            export_text += f"Spec Reference: {result['infraction']['spec_ref']}\n"
            export_text += f"Page Number: {result['infraction']['page_num']}\n"
            export_text += f"Infraction: {result['infraction']['text']}\n"
            export_text += f"Auditor's Reasoning: {result['infraction']['reasoning']}\n\n"
            export_text += "ANALYSIS:\n"
            export_text += result['analysis'] + "\n"
            export_text += "="*50 + "\n"
        
        st.download_button(
            label="Download Analysis Report",
            data=export_text,
            file_name="qa_audit_analysis.txt",
            mime="text/plain"
        )

# Instructions panel
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    ### How to Use:
    
    1. **Upload Spec Book**: Upload the specification standards document (PDF or DOCX)
    2. **Click 'Index Spec Book'**: This will process and index the spec book
    3. **Upload QA Audit**: Upload the audit document with infractions
    4. **Click 'Analyze Audit'**: The AI will analyze each infraction
    
    ### What the Analyzer Does:
    - Extracts infractions from page 2+ under "Flagged Items"
    - Identifies spec references and page numbers
    - Cross-references with the spec book
    - Provides confidence ratings (0-100%)
    - Determines if infractions can be overturned
    - Gives detailed reasoning for each decision
    
    ### Tips:
    - Ensure audit document has clear "Flagged Items" section
    - Spec references should be clearly marked
    - The more detailed the spec book, the better the analysis
    """) 
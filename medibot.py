import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_community.llms import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
# Path to the FAISS vector store directory
DB_FAISS_PATH = "vectorstore/db_faiss"
# Load FAISS vector store once using Streamlit cache
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db
# Custom prompt template setup
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )
# Streamlit app
def main():
    # 🌟 Page configuration for professional layouts
    st.set_page_config(
        page_title="MediBot - AI Medical Assistant",
        page_icon="🩺",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    # 🎨 Custom Premium Theme Styling Injection
    st.markdown("""
        <style>
        /* Import premium modern font */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
        /* Global style definitions */
        .stApp {
            background: radial-gradient(circle at top right, #0B0F19 0%, #070A13 100%) !important;
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
            color: #E2E8F0 !important;
        }
        /* Top Header styling */
        [data-testid="stHeader"] {
            background: rgba(11, 15, 25, 0.6) !important;
            backdrop-filter: blur(12px) !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(11, 15, 25, 0.95) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        
        /* Chat Message bubble adjustments */
        [data-testid="stChatMessage"] {
            background-color: rgba(30, 41, 59, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 18px !important;
            padding: 16px 20px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.12) !important;
            margin-bottom: 12px !important;
            backdrop-filter: blur(8px) !important;
        }
        /* User Message specific background */
        [data-testid="stChatMessage"][aria-label="user"] {
            background-color: rgba(99, 102, 241, 0.08) !important;
            border: 1px solid rgba(99, 102, 241, 0.25) !important;
        }
        /* Assistant Message specific background */
        [data-testid="stChatMessage"][aria-label="assistant"] {
            background-color: rgba(14, 165, 233, 0.06) !important;
            border: 1px solid rgba(14, 165, 233, 0.25) !important;
        }
        /* Styling for the sticky chat input */
        [data-testid="stChatInput"] {
            border-radius: 24px !important;
            border: 1px solid rgba(14, 165, 233, 0.3) !important;
            background-color: rgba(11, 15, 25, 0.85) !important;
            backdrop-filter: blur(16px) !important;
            box-shadow: 0 -8px 32px rgba(0, 0, 0, 0.25) !important;
        }
        /* Textarea input font color */
        [data-testid="stChatInput"] textarea {
            color: #F8FAFC !important;
            font-size: 0.95rem !important;
        }
        /* Hide generic Streamlit UI items for cleaner white-label look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* Welcome panel styling */
        .welcome-panel {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 28px;
            margin-bottom: 28px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
        }
        /* Suggestions card layout */
        .suggestion-card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 14px;
            padding: 18px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            height: 100%;
        }
        .suggestion-card:hover {
            border-color: rgba(14, 165, 233, 0.35);
            background: rgba(14, 165, 233, 0.04);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(14, 165, 233, 0.08);
        }
        </style>
    """, unsafe_allow_html=True)
    # 🖥️ Sidebar content
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <span style="font-size: 3.5rem; filter: drop-shadow(0 0 12px rgba(14, 165, 233, 0.4));">🩺</span>
            <h2 style="margin: 15px 0 5px 0; color: #FFFFFF; font-weight: 700; font-family: 'Plus Jakarta Sans';">MediBot Core</h2>
            <p style="margin: 0; color: #94A3B8; font-size: 0.85rem; font-weight: 500;">Clinical Intelligence Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.05);' />", unsafe_allow_html=True)
        
        # Diagnostics block
        st.markdown("### ⚙️ System Diagnosis")
        st.markdown("""
        * **Engine Model:** `Llama-3.3-70b`
        * **RAG Pipeline:** `LangChain & FAISS`
        * **Embeddings:** `all-MiniLM-L6-v2`
        * **Vector Cache:** `Operational`
        """)
        
        st.markdown("<hr style='border-color: rgba(255,255,255,0.05);' />", unsafe_allow_html=True)
        
        # Reset conversation trigger
        if st.button("🔄 Reset Conversation History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("<hr style='border-color: rgba(255,255,255,0.05);' />", unsafe_allow_html=True)
        
        # Medical Disclaimer
        st.markdown("""
        <div style="background-color: rgba(239, 68, 68, 0.06); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 12px; padding: 14px; font-size: 0.78rem; line-height: 1.45; color: #FCA5A5; font-family: 'Plus Jakarta Sans';">
            <strong>⚠️ Medical Disclaimer:</strong><br>
            This system provides literature references and answers using automated retrieval. It does not provide clinical diagnosis or direct medical advice. Consult a healthcare provider for any diagnostic or medical treatment queries.
        </div>
        """, unsafe_allow_html=True)
    # 🏢 Main Header layout
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 18px; margin-bottom: 25px; border-bottom: 1px solid rgba(255,255,255,0.05);">
        <div style="display: flex; align-items: center;">
            <span style="font-size: 2.2rem; margin-right: 14px; filter: drop-shadow(0 0 8px rgba(14,165,233,0.3));">💬</span>
            <div>
                <h1 style="margin: 0; font-size: 1.7rem; font-weight: 800; color: #FFFFFF; font-family: 'Plus Jakarta Sans'; letter-spacing: -0.5px;">
                    Medi<span style="color: #0EA5E9;">Bot</span>
                </h1>
                <p style="margin: 0; font-size: 0.75rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px;">
                    Clinical RAG Search Engine
                </p>
            </div>
        </div>
        <div style="display: flex; align-items: center; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 20px; padding: 5px 12px;">
            <span style="width: 8px; height: 8px; background-color: #10B981; border-radius: 50%; display: inline-block; margin-right: 8px; box-shadow: 0 0 8px #10B981;"></span>
            <span style="font-size: 0.72rem; font-weight: 700; color: #34D399; text-transform: uppercase; letter-spacing: 0.5px;">LLaMA-3.3 Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Initialize messages list
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    # 🤝 Welcome Cards if no conversation has taken place
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-panel">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 2.4rem; margin-right: 16px;">🩺</span>
                <div>
                    <h3 style="margin: 0; color: #FFFFFF; font-size: 1.5rem; font-weight: 700; font-family: 'Plus Jakarta Sans';">Welcome to MediBot Intelligence Portal</h3>
                    <p style="margin: 0; color: #94A3B8; font-size: 0.88rem;">Empowered by Retrieval-Augmented Generation (RAG)</p>
                </div>
            </div>
            <p style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.6; margin-top: 14px;">
                Enter a question below, and MediBot will dynamically query our localized clinical FAISS vector index to pull relevant medical studies, pharmacology profiles, or diagnostic details, generating a contextual response referencing standard medical literature.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h4 style='color: #F8FAFC; margin-bottom: 16px; font-weight: 600; font-family: \"Plus Jakarta Sans\";'>💡 Sample Clinical Queries</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="suggestion-card">
                <h5 style="color: #38BDF8; margin: 0 0 6px 0; font-size: 0.98rem; font-weight: 600;">🩺 Symptoms Guidance</h5>
                <p style="color: #94A3B8; margin: 0; font-size: 0.85rem; line-height: 1.45;">
                    "What are the diagnostic indicators and typical symptom profiles of Type 2 Diabetes?"
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="suggestion-card">
                <h5 style="color: #38BDF8; margin: 0 0 6px 0; font-size: 0.98rem; font-weight: 600;">💊 Drug Interactions</h5>
                <p style="color: #94A3B8; margin: 0; font-size: 0.85rem; line-height: 1.45;">
                    "Describe standard side effects and contraindicated combinations for Lisinopril medication."
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="suggestion-card">
                <h5 style="color: #14B8A6; margin: 0 0 6px 0; font-size: 0.98rem; font-weight: 600;">🔬 Diagnostic Procedures</h5>
                <p style="color: #94A3B8; margin: 0; font-size: 0.85rem; line-height: 1.45;">
                    "What criteria guide clinician workflows when sorting diagnostic results for acute coronary syndromes?"
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="suggestion-card">
                <h5 style="color: #14B8A6; margin: 0 0 6px 0; font-size: 0.98rem; font-weight: 600;">🧬 Literature Inquiry</h5>
                <p style="color: #94A3B8; margin: 0; font-size: 0.85rem; line-height: 1.45;">
                    "Summarize standard medical literature research findings on vaccine antibody retention rates over time."
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    # 💬 Render chat history
    for message in st.session_state.messages:
        role = message['role']
        avatar = "👤" if role == "user" else "🩺"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message['content'])
    # ⌨️ Accept chat query input
    prompt = st.chat_input("Pass your medical query here...")
    if prompt:
        # Display user query
        with st.chat_message('user', avatar="👤"):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        # Prompt template for RAG
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don’t know the answer, say you don’t know. Don’t try to make up an answer.
        Only use information from the context provided.
        Context: {context}
        Question: {question}
        Answer:
        """
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load clinical vector database.")
                return
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
                }
            )
            # Elegant clinical processing spinner
            with st.spinner("Analyzing medical knowledge base..."):
                response = qa_chain.invoke({"query": prompt})
                
            result = response["result"]
            sources = response["source_documents"]
            # Parse referenced document files and format nicely
            source_lines = []
            for doc in sources:
                source_path = doc.metadata.get('source', 'Unknown')
                # Grab base filename to make paths elegant
                source_name = source_path.split("/")[-1].split("\\")[-1]
                page = doc.metadata.get('page') or doc.metadata.get('page_number') or 'N/A'
                source_lines.append(f"📄 **{source_name}** | 📑 Page: {page}")
            # 🔒 Hide references if answer is fallback
            if "I don’t know" in result or "I couldn’t find relevant information" in result:
                result_to_show = result
            else:
                sources_text = "\n".join([f"- {line}" for line in source_lines])
                result_to_show = f"{result}\n\n💡 **Sources & Reference Material:**\n{sources_text}"
            # Display response in assistant bubble
            with st.chat_message("assistant", avatar="🩺"):
                st.markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
        except Exception as e:
            st.error("🩺 An error occurred during diagnosis. See trace details below:")
            import traceback
            st.code(traceback.format_exc())
if __name__ == "__main__":
    main()
import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

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

# Load Hugging Face endpoint
def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
    repo_id=huggingface_repo_id,
    temperature=0.5,
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512,
    return_full_text=False
)


    return llm

# Streamlit app
def main():
    st.title("💬 MediBot - Your Medical Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your medical query here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
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

        # Hugging Face credentials
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load vector store.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({"query": prompt})
            result = response["result"]
            sources = response["source_documents"]

            # Format the sources with page numbers
            source_lines = []
            for doc in sources:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page') or doc.metadata.get('page_number') or 'N/A'
                source_lines.append(f"- 📄 **{source}** | 📑 Page: {page}")

            # 🔒 Hide sources if answer is a fallback response
            if "I don’t know" in result or "I couldn’t find relevant information" in result:
                result_to_show = result
            else:
                result_to_show = result + "\n\n📝 **Sources:**\n" + "\n".join(source_lines)

            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    main()
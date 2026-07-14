import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# ✅ Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in .env")

# ✅ Set token for Hugging Face library
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# ✅ Function to load the LLM from Hugging Face
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_new_tokens=512
    )
    return llm

# ✅ Load FAISS index and embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(
    "vectorstore/db_faiss",  # ✅ Corrected path
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 2})

# ✅ Custom prompt template
prompt_template = """Use the following context to answer the question.
If you don't know the answer, just say you don't know — don't make anything up.

Context:
{context}

Question:
{question}

Helpful Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ✅ Load LLM
llm = load_llm("mistralai/Mistral-7B-Instruct-v0.3")  # Or another supported model

# ✅ Setup RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# ✅ User interaction
if __name__ == "__main__":
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({'query': user_query})
    print("\nResponse:\n", response["result"])
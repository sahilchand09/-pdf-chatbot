import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# --- CSS - Design ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 15px; }
    h1 { text-align: center; font-size: 42px !important; }
    .subtitle { text-align: center; color: #888; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

st.title("📄 PDF Chatbot")
st.markdown('<p class="subtitle">Apne PDF se kuch bhi pucho, AI jawab dega!</p>', unsafe_allow_html=True)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY .env file me nahi mili!")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("1. PDF upload karo")
    st.write("2. Neeche question pucho")
    st.write("3. AI PDF se jawab dega")
    st.divider()
    uploaded_file = st.file_uploader("📁 PDF Upload Karo", type="pdf")
    if st.button("🗑️ Chat Clear Karo"):
        st.session_state.messages = []
        st.rerun()

# --- Chat History Start ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file is not None and st.session_state.vectorstore is None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    with st.spinner("PDF padh raha hu..."):
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    st.sidebar.success(f"✅ PDF Ready! {len(chunks)} chunks bane")

# --- Purani Chat Dikhana ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Naya Question ---
if st.session_state.vectorstore is None:
    st.info("👈 Pehle Sidebar se PDF Upload Karo")
else:
    question = st.chat_input("Apna Question Pucho...")

    if question:
        # User ka message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # AI ka jawab
        with st.chat_message("assistant"):
            with st.spinner("Soch raha hu..."):
                retriever = st.session_state.vectorstore.as_retriever()
                relevant_docs = retriever.invoke(question)
                context = "\n\n".join([d.page_content for d in relevant_docs])

                llm = ChatGroq(
                    model_name="openai/gpt-oss-20b",
                    temperature=0,
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )

                prompt = ChatPromptTemplate.from_template("""
                Tum ek helpful assistant ho. Neeche diye gaye context se hi jawab do, Hindi + English mix me, simple language me.
                Agar jawab context me nahi hai to bol do "Mujhe is PDF me ye nahi mila".

                Context:
                {context}

                Question: {question}
                Answer:
                """)

                messages = prompt.format_messages(context=context, question=question)
                response = llm.invoke(messages)
                st.write(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
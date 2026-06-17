import os
import requests
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. CONFIGURAÇÕES INICIAIS E UI (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Projeto Athena - RAG", page_icon="🏛️", layout="wide")

# Injeção de CSS para identidade visual (Lavanda/Lilás) e painel lateral de emergência
st.markdown("""
    <style>
    .stApp { background-color: #fcfbfe; }
    .css-1d391kg { background-color: #f3effa; } /* Barra lateral */
    .emergency-card {
        background: linear-gradient(135deg, #ff758c 0%, #ff7eb3 100%);
        color: white; padding: 15px; border-radius: 10px;
        text-align: center; font-weight: bold; margin-bottom: 20px;
    }
    .chat-athena { color: #5b21b6; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

# Inicialização do Histórico do Chat no State
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. SEGREDO / AMBIENTE (HUGGING FACE)
# ==========================================
# Cadastrar 'HF_API_TOKEN' nos Secrets do Streamlit Cloud
if "st.secrets" in globals() and "HF_API_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_API_TOKEN"]
else:
    HF_TOKEN = os.getenv("HF_API_TOKEN", "")

# Endpoint da API do Llama-3 na Hugging Face
LLM_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# ==========================================
# 3. MOTOR DO RAG (LANGCHAIN + FAISS)
# ==========================================
@st.cache_resource(show_spinner=False)
def inicializar_base_conhecimento():
    """Varre a pasta local, fragmenta os PDFs e indexa no FAISS."""
    diretorio_conhecimento = "conhecimento"
    if not os.path.exists(diretorio_conhecimento):
        os.makedirs(diretorio_conhecimento)
        
    arquivos_pdf = [
        os.path.join(diretorio_conhecimento, f) 
        for f in os.listdir(diretorio_conhecimento) 
        if f.lower().endswith('.pdf')
    ]
    
    if not arquivos_pdf:
        return None

    todos_os_documentos = []
    for caminho_pdf in arquivos_pdf:
        try:
            loader = PyPDFLoader(caminho_pdf)
            todos_os_documentos.extend(loader.load())
        except Exception as e:
            print(f"Erro ao ler arquivo {caminho_pdf}: {e}")

    # Fragmentação Semântica (Tamanho

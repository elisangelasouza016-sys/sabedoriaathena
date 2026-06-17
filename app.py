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

    # Fragmentação Semântica (Tamanho ideal para artigos de lei)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    fragmentos = text_splitter.split_documents(todos_os_documentos)

    # Inicialização dos Embeddings via Módulo Clássico da Community
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db_vetorial = FAISS.from_documents(fragmentos, embeddings)
    return db_vetorial

# Inicializa o banco de dados vetorial
with st.spinner("Construindo e indexando a base de conhecimento jurídica..."):
    vector_db = inicializar_base_conhecimento()

# ==========================================
# 4. BARRA LATERAL (INFORMAÇÃO E SOCORRO)
# ==========================================
with st.sidebar:
    st.title("🏛️ Ecossistema Athena")
    st.subheader("IA para Orientação e Proteção da Mulher")
    st.markdown("---")
    
    # Card de Emergência de Alta Visibilidade
    st.markdown("""
        <div class="emergency-card">
            🚨 CANAIS DE EMERGÊNCIA<br>
            <span style="font-size: 20px;">LIGUE 180 (Mulher)</span><br>
            <span style="font-size: 20px;">LIGUE 190 (Polícia)</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📚 Documentos Ativos no RAG:")
    
    if vector_db:
        st.success("🟢 Base de Conhecimento Pronta!")
        for f in os.listdir("conhecimento"):
            if f.lower().endswith('.pdf'):
                st.caption(f"📖 `{f}`")
    else:
        st.warning("⚠️ Nenhum arquivo PDF encontrado na pasta `conhecimento/`.")

# ==========================================
# 5. ENGENHARIA DE PROMPT (ATHENA PERSONA)
# ==========================================
PROMPT_TEMPLATE = """
Você é a Athena, uma assistente virtual jurídica e de acolhimento altamente especializada nos direitos das mulheres e redes de apoio.
Sua persona deve ser firme, justa, acolhedora e extremamente profissional, traduzindo termos jurídicos complexos para uma linguagem acessível.

Use estritamente os fragmentos de contexto fornecidos abaixo para responder à pergunta da usuária. Se a resposta não puder ser encontrada no contexto, diga de forma acolhedora que não possui essa informação oficial no momento, mas reforce os canais de apoio gerais.

Regra de Segurança Física: Se o relato da usuária indicar perigo iminente ou violência doméstica recente, insira OBRIGATORIAMENTE no topo da resposta a instrução de ligar para o 180 ou 190.

Contexto Recuperado:
{contexto}

Pergunta da Usuária:
{pergunta}

Resposta da Athena (em português do Brasil, estruturada e acolhedora):
"""
prompt_athena = PromptTemplate(input_variables=["contexto", "pergunta"], template=PROMPT_TEMPLATE)

# ==========================================
# 6. INTERFACE PRINCIPAL DO CHAT
# ==========================================
st.title("🏛️ Conversar com a Athena")
st.write("Consulte de forma segura e privada leis, cartilhas e telefones de suporte.")

# Renderiza mensagens anteriores do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Caixa de Entrada de Texto Padrão do Chat
pergunta_usuario = st.chat_input("Digite aqui sua dúvida jurídica...")

# ==========================================
# 7. EXECUÇÃO DO PIPELINE RAG + INFERÊNCIA
# ==========================================
if pergunta_usuario:
    # Adiciona e exibe a mensagem da usuária
    st.session_state.messages.append({"role": "user", "content": pergunta_usuario})
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)

    contexto_recuperado = ""
    if vector_db:
        # Busca por similaridade semântica
        docs_similares = vector_db.similarity_search(pergunta_usuario, k=6)
        
        # Algoritmo de Priorização: Força arquivos de telefones para o topo
        docs_ordenados = sorted(
            docs_similares, 
            key=lambda x: 0 if "telefones" in str(x.metadata.get("source", "")).lower() else 1
        )
        contexto_recuperado = "\n\n".join([doc.page_content for doc in docs_ordenados])

    # Injeta contexto e pergunta no template
    prompt_finalizado = prompt_athena.format(contexto=contexto_recuperado, pergunta=pergunta_usuario)

    # Bloco de geração da resposta
    with st.chat_message("assistant"):
        placeholder_resposta = st.empty()
        with st.spinner("Athena está analisando as bases legais..."):
            try:
                payload = {
                    "inputs": prompt_finalizado,
                    "parameters": {
                        "temperature": 0.5,       
                        "max_new_tokens": 800,    
                        "return_full_text": False
                    }
                }
                
                resposta_llm = requests.post(LLM_API_URL, headers=headers, json=payload)
                
                if resposta_llm.status_code == 200:
                    dados_retorno = resposta_llm.json()
                    if isinstance(dados_retorno, list):
                        texto_resposta = dados_retorno[0].get("generated_text", "")
                    else:
                        texto_resposta = dados_retorno.get("generated_text", "")
                        
                    # Limpa repetições indesejadas do prompt
                    texto_resposta = texto_resposta.replace(prompt_finalizado, "").strip()
                    
                    # Exibe e armazena a resposta final
                    placeholder_resposta.markdown(texto_resposta)
                    st.session_state.messages.append({"role": "assistant", "content": texto_resposta})
                    
                else:
                    st.error(f"Erro de conexão com o oráculo (Código {resposta_llm.status_code})")
            except Exception as e:
                st.error(f"Erro no pipeline de inferência: {e}")

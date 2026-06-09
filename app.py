import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Importações para Processamento de Documentos
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Importações de Persona e Limpeza da Athena
from athena import ATHENA_SYSTEM_PROMPT, limpar_tags_llama3

# Carregar variáveis de ambiente
load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# --- CONFIGURAÇÃO DO CONHECIMENTO (RAG) ---
CACHE_DIR = "vector_cache"
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index")

@st.cache_resource # Mantém o objeto FAISS carregado em memória na sessão do app
def carregar_conhecimento():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Se o cache do FAISS já existe localmente, carrega em milissegundos
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            vector_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            return vector_db, True # Retorna o banco e flag indicando que veio do cache
        except Exception as e:
            st.error(f"Erro ao carregar o cache do FAISS: {e}. Reconstruindo base...")
            
    # Caso contrário, inicializa e indexa os arquivos da pasta 'conhecimento'
    if not os.path.exists("conhecimento"):
        os.makedirs("conhecimento")
        return None, False

    # Carregar PDFs
    loader = DirectoryLoader('conhecimento/', glob="*.pdf", loader_cls=PyPDFLoader)
    try:
        docs = loader.load()
    except Exception as e:
        st.error(f"Erro ao ler arquivos da pasta 'conhecimento': {e}")
        return None, False
        
    if not docs:
        return None, False

    # Dividir texto em pedaços de 700 caracteres com 100 de sobreposição
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Criar Banco de Dados de busca (FAISS)
    try:
        vector_db = FAISS.from_documents(chunks, embeddings)
        # Persistir fisicamente para acelerar os próximos carregamentos
        os.makedirs(CACHE_DIR, exist_ok=True)
        vector_db.save_local(FAISS_INDEX_PATH)
        return vector_db, False # Retorna o banco e indica que foi gerado do zero
    except Exception as e:
        st.error(f"Erro ao gerar a base vetorial FAISS: {e}")
        return None, False

# Inicializar Base de Dados RAG
db, carregado_do_cache = carregar_conhecimento()

# --- CONFIGURAÇÃO DA PÁGINA E ESTILO VISUAL PREMIUM ---
st.set_page_config(page_title="Athena: Orientação e Direitos das Mulheres", page_icon="🏛️", layout="wide")

# CSS Customizado para Estética Premium (Tons escuros, bronze, ouro e azul petróleo)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@400;600;700&display=swap');
    
    /* Reset e Variáveis de Fonte - Fundo Claro Lilás */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #F6F4F9 !important;
        color: #2D1A3F !important;
    }
    
    /* Forçar cores do corpo do texto e listas */
    .stApp p, .stApp li, .stApp span, .stApp label, .stApp td, .stApp th {
        color: #2D1A3F !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #6D28D9 !important; /* Roxo / Lilás Escuro */
        letter-spacing: 0.5px;
    }
    
    /* Estilização da Barra Lateral - Tons Lilás Claro */
    section[data-testid="stSidebar"] {
        background-color: #EFEBF4 !important;
        border-right: 1px solid #D8CFE5 !important;
        box-shadow: 2px 0 10px rgba(109, 40, 217, 0.05) !important;
    }
    
    section[data-testid="stSidebar"] hr {
        border-top: 1px solid #D8CFE5 !important;
    }
    
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] li {
        color: #3C2258 !important;
    }
    
    /* Customização dos Balões do Chat - Fundo Branco e Sombras Suaves */
    .stChatMessage {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        border: 1px solid #E5DFEE !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 8px rgba(109, 40, 217, 0.03) !important;
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        border-color: #C7B9DC !important;
        box-shadow: 0 4px 12px rgba(109, 40, 217, 0.08) !important;
    }
    
    /* Balão do usuário - Roxo Claro */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #FFFFFF !important;
        border-left: 4px solid #C084FC !important;
    }
    
    .stChatMessage[data-testid="stChatMessageUser"] p {
        color: #2D1A3F !important;
    }
    
    /* Balão da assistente (Athena) - Lavanda Suave */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #F5F3FF !important;
        border-left: 4px solid #7C3AED !important;
    }
    
    .stChatMessage[data-testid="stChatMessageAssistant"] p {
        color: #2D1A3F !important;
    }
    
    /* Caixa de entrada do chat - Fundo Branco e Bordas Lilás */
    .stChatInput textarea {
        background-color: #FFFFFF !important;
        color: #2D1A3F !important;
        border: 1px solid #D1C7DF !important;
        border-radius: 8px !important;
    }
    
    .stChatInput textarea:focus {
        border-color: #7C3AED !important;
        box-shadow: 0 0 8px rgba(124, 58, 237, 0.2) !important;
    }
    
    /* Botões Customizados */
    .stButton>button {
        background-color: #FFFFFF !important;
        color: #7C3AED !important;
        border: 1px solid #7C3AED !important;
        font-family: 'Outfit', sans-serif !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        width: 100%;
        font-weight: 500 !important;
    }
    
    .stButton>button:hover {
        background-color: #7C3AED !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 10px rgba(124, 58, 237, 0.3) !important;
        transform: translateY(-1px);
    }
    
    /* Card de Alerta/Emergência na Sidebar - Paleta Lilás/Rosa Suave e Texto de Alta Visibilidade */
    .emergency-card {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%) !important;
        border: 1px solid #FDA4AF !important;
        border-radius: 8px;
        padding: 14px;
        margin-top: 15px;
        color: #9F1239 !important;
        box-shadow: 0 4px 10px rgba(225, 29, 72, 0.05) !important;
    }
    
    .emergency-card h4 {
        color: #9F1239 !important;
        margin-top: 0px;
        margin-bottom: 8px;
        font-size: 1.1em;
    }
    
    .emergency-card p, .emergency-card b {
        font-size: 0.9em;
        margin-bottom: 4px;
        line-height: 1.4;
        color: #9F1239 !important;
    }
    
    /* Indicadores de status */
    .status-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 500;
        text-align: center;
        background-color: #D1FAE5 !important;
        color: #065F46 !important;
        border: 1px solid #A7F3D0 !important;
    }
    
    .status-badge-cache {
        background-color: #EDE9FE !important;
        color: #5B21B6 !important;
        border: 1px solid #DDD6FE !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÃO DA SIDEBAR ---
with st.sidebar:
    st.markdown("<h1>Athena 🏛️</h1>", unsafe_allow_html=True)
    st.markdown("*Sabedoria e Direitos da Mulher*")
    st.markdown("---")
    
    # Seção sobre a Assistente
    st.markdown("### Sobre a Athena")
    st.markdown(
        "Athena é uma inteligência artificial inspirada na deusa da sabedoria e justiça. "
        "Seu propósito é fornecer orientações claras, acolhedoras e fundamentadas em leis "
        "para ajudar mulheres a entenderem e exercerem seus direitos."
    )
    
    st.markdown("---")
    
    # Status da Base de Conhecimento RAG
    st.markdown("### Base de Conhecimento (RAG)")
    if db:
        # Obter arquivos na pasta
        arquivos = [f for f in os.listdir("conhecimento") if f.endswith(".pdf")]
        num_arquivos = len(arquivos)
        
        # Badge de status do cache
        if carregado_do_cache:
            st.markdown('<span class="status-badge status-badge-cache">⚡ Pronto (do cache)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge">🟢 Pronto (indexado agora)</span>', unsafe_allow_html=True)
            
        st.markdown(f"**{num_arquivos} documentos PDF** integrados com sucesso.")
        
        # Lista expansível dos documentos
        with st.expander("Ver arquivos indexados"):
            for arq in arquivos:
                st.markdown(f"- 📄 `{arq}`")
    else:
        st.markdown("⚠️ Nenhuma base de dados encontrada em `conhecimento/`.")
        
    st.markdown("---")
    
    # Card de Contatos de Emergência de Alta Visibilidade
    st.markdown("""
        <div class="emergency-card">
            <h4>🚨 Apoio & Emergência</h4>
            <p>Se você ou alguém que conhece está sofrendo qualquer tipo de violência física ou psicológica, peça ajuda:</p>
            <p><b>• Central de Atendimento:</b> Ligue <b>180</b> (Gratuito e sigiloso)</p>
            <p><b>• Emergência Policial:</b> Ligue <b>190</b></p>
            <p><b>• Apoio Jurídico:</b> Defensoria Pública da sua região</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Botão de reinicialização da conversa
    if st.button("Reiniciar Conversa"):
        st.session_state.messages = [{"role": "assistant", "content": "Saudações. Sou Athena. Em que posso orientá-la hoje?"}]
        st.rerun()

# --- ÁREA CENTRAL DO CHAT ---
st.markdown("<h3>Orientação Jurídica Inteligente</h3>", unsafe_allow_html=True)
st.markdown(
    "Utilize a Athena para tirar dúvidas sobre a Lei Maria da Penha, direitos da gestante, "
    "direitos trabalhistas da mulher, empreendedorismo, contatos de Delegacias Especializadas do RN e mais. Suas perguntas serão cruzadas "
    "com a nossa biblioteca de leis oficiais para trazer respostas fundamentadas."
)

# Inicializar histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Saudações. Sou Athena. Em que posso orientá-la hoje?"}]

# Renderizar mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Função para realizar inferência via Hugging Face Router API (OpenAI compatível)
def query_hugging_face(messages):
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.5
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=25)
        if response.status_code == 200:
            res_json = response.json()
            if 'choices' in res_json and len(res_json['choices']) > 0:
                return res_json['choices'][0]['message']['content']
        else:
            return f"Erro de conexão com o oráculo (Código {response.status_code}): {response.text}"
    except Exception as e:
        return f"Falha de conexão com os servidores da IA: {str(e)}"
    return "Erro: não foi possível gerar uma resposta no momento."

# --- FLUXO DE ENTRADA DO USUÁRIO ---
if prompt := st.chat_input("Sua pergunta sobre direitos, leis ou redes de apoio..."):
    # Salvar a pergunta do usuário no histórico e renderizar na tela
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- RECUPERAÇÃO DE CONTEXTO (RAG) COM PRIORIZAÇÃO ---
    contexto = ""
    if db:
        # Busca k=6 documentos similares para cruzar mais referências
        docs_relacionados = db.similarity_search(prompt, k=6)
        
        # Separar e ordenar os documentos: telefones_uteis_estado.pdf sempre no topo
        prioritarios = []
        outros = []
        for doc in docs_relacionados:
            source = doc.metadata.get("source", "").lower()
            if "telefones_uteis_estado.pdf" in source:
                prioritarios.append(doc)
            else:
                outros.append(doc)
                
        docs_ordenados = prioritarios + outros
        contexto = "\n\n---\n\n".join([doc.page_content for doc in docs_ordenados])

    # --- GESTÃO DE MEMÓRIA (HISTÓRICO RECENTE) ---
    # Extrair no máximo as últimas 3 trocas de turnos completas do histórico de mensagens anteriores
    # st.session_state.messages[-1] é a pergunta que acabamos de adicionar.
    # O histórico anterior é st.session_state.messages[:-1]. Pegamos os últimos 6 itens dele.
    historico_recente = st.session_state.messages[:-1]
    if len(historico_recente) > 6:
        historico_recente = historico_recente[-6:]

    # --- INSTRUÇÃO DE SISTEMA INCORPORANDO CONTEXTO ---
    system_instruction = (
        f"{ATHENA_SYSTEM_PROMPT}\n\n"
        "--- CONTEXTO DE BUSCA VETORIAL DA BIBLIOTECA DE LEIS ---\n"
        f"{contexto if contexto else 'Nenhum documento específico encontrado para esta busca.'}\n"
        "--- FIM DO CONTEXTO ---"
    )

    # --- CONSTRUÇÃO DOS DIÁLOGOS NO PADRÃO OPENAI ---
    api_messages = [{"role": "system", "content": system_instruction}]
    for msg in historico_recente:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    api_messages.append({"role": "user", "content": prompt})

    # --- PROCESSAMENTO E INFERÊNCIA ---
    with st.chat_message("assistant"):
        with st.spinner("Consultando leis, decretos e oráculos..."):
            resposta_bruta = query_hugging_face(api_messages)
            # Limpar a resposta gerada para evitar resíduos das tags do Llama-3
            resposta_limpa = limpar_tags_llama3(resposta_bruta)
            st.markdown(resposta_limpa)
    
    # Adicionar resposta final ao histórico da sessão
    st.session_state.messages.append({"role": "assistant", "content": resposta_limpa})

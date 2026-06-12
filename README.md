# 🏛️ Projeto Athena — IA Generativa e Direitos da Mulher

O **Ecossistema Athena** é uma aplicação web assistencial baseada na arquitetura **RAG (Retrieval-Augmented Generation)**. O sistema atua como uma especialista jurídica e rede de apoio virtual, democratizando o acesso a direitos fundamentais através da tradução de legislações complexas em diálogos interativos, acessíveis e psicologicamente seguros.

Acesse o MVP aqui (https://sabedoriaathena.streamlit.app/)

---

## 🛠️ Pilha Tecnológica (Pilha Obrigatória)
* **Interface (UI/UX):** (https://sabedoriaathena.streamlit.app/)(com estilização customizada via CSS nativo).
* **Orquestração RAG:** [LangChain](https://www.langchain.com/) (`PyPDFLoader`, `RecursiveCharacterTextSplitter`, `FAISS`).
* **Representação Vetorial:** [HuggingFace Embeddings](https://huggingface.co/) (`all-MiniLM-L6-v2`).
* **Modelo de Linguagem (LLM):** `Meta-Llama-3-8B-Instruct` (via Hugging Face Router API).

---

## ⚙️ Arquitetura e Fluxo do RAG
1. **Ingestão Dinâmica:** Varredura nativa em Python (`os.listdir`) de arquivos normativos em formato `.pdf` dentro do diretório `conhecimento/`.
2. **Fragmentação Semântica:** Divisão dos textos em blocos de 700 caracteres (`chunk_size=700`, `chunk_overlap=100`) para preservar a coesão de artigos de lei.
3. **Indexação:** Vetorização e armazenamento em base de dados local **FAISS**, com persistência física em cache local.
4. **Recuperação Inteligente:** Filtro por similaridade de cosseno (`k=6`) com algoritmo de priorização para colocar canais de emergência (`telefones_uteis_estado.pdf`) sempre no topo do contexto.
5. **Inferência:** LLM parametrizado com **Temperatura fixa em 0.5** para mitigação absoluta de alucinações e garantia de rigor factual.

---

## 🚀 Como Executar Localmente

1. **Clonar o Repositório:**
   ```bash
   git clone [https://github.com/](https://github.com/)[seu-usuario]/[seu-repositorio].git
   cd [seu-repositorio]

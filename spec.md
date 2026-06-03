

### 1. Arquitetura do Sistema: RAG (Retrieval-Augmented Generation)

A Athena não utiliza apenas um modelo de linguagem pré-treinado (LLM). Ela implementa uma arquitetura de **Geração Aumentada por Recuperação**. Isso significa que o sistema possui um ciclo de vida em duas etapas:

* **Etapa de Indexação (Offline/Startup):** O sistema lê os documentos PDF na pasta `/conhecimento`, fragmenta-os em blocos de texto (*chunks*) e converte esses blocos em vetores numéricos (embeddings).
* **Etapa de Recuperação (Runtime):** Quando a utilizadora faz uma pergunta, o sistema converte a pergunta em vetor, procura no banco de dados os fragmentos mais similares e envia-os como contexto para o LLM.

---

### 2. Stack Tecnológica (Componentes)

* **Interface de Utilizador (Frontend):** **Streamlit**. Uma biblioteca Python que permite a criação de interfaces web reativas para ciência de dados, gerindo o estado da sessão (histórico do chat).
* **Orquestração de Dados:** **LangChain**. Atua como a "cola" do projeto, gerindo os carregadores de documentos (`DirectoryLoader`), os divisores de texto (`RecursiveCharacterTextSplitter`) e a lógica de busca.
* **Motor de Busca Vetorial:** **FAISS (Facebook AI Similarity Search)**. Uma biblioteca de código aberto para busca de similaridade densa, que permite encontrar rapidamente informações em grandes volumes de texto.
* **Modelo de Embeddings:** **sentence-transformers/all-MiniLM-L6-v2**. Um modelo leve que roda localmente para transformar texto em vetores de 384 dimensões.
* **Modelo de Linguagem (Inference):** **Meta-Llama-3-8B-Instruct**. Acedido via API de Inferência do Hugging Face. É o modelo responsável por ler o contexto e gerar a resposta final em linguagem natural.

---

### 3. Fluxo de Dados Técnico (Step-by-Step)

1. **Processamento de Texto:** O arquivo PDF é lido e dividido em partes de 700 caracteres com uma sobreposição (*overlap*) de 100 caracteres. A sobreposição garante que o contexto não seja cortado abruptamente entre blocos.
2. **Vetorização:** Cada bloco passa pelo modelo de *embeddings*. O resultado é armazenado num índice FAISS.
3. **Prompt Engineering:** O sistema utiliza um template de prompt estruturado para o Llama-3, injetando instruções de sistema (System Message), o contexto recuperado dos PDFs e a pergunta da utilizadora.
4. **Inferência via API:** O prompt final é enviado via POST request (JSON) para os servidores do Hugging Face com os seguintes hiperparâmetros:
* **Temperature (0.5/0.6):** Equilíbrio entre precisão e fluidez.
* **Max New Tokens (600+):** Garante que a resposta não seja cortada.


5. **Pós-processamento:** A resposta é limpa de tokens especiais (como `<|eot_id|>`) e exibida no Streamlit.

---

### 4. Gestão de Memória e Estado

Como as APIs de LLM são apátridas (*stateless*), o MVP utiliza o `st.session_state` do Streamlit para armazenar a lista de mensagens da conversa. A cada nova interação, o sistema "relembra" o modelo das últimas 3 trocas de mensagens, enviando-as de volta no prompt para manter a coerência do diálogo.

---

### 5. Considerações de Segurança (Privacy by Design)

* **Tokens de Ambiente:** Uso de arquivos `.env` para que as chaves de API não fiquem expostas no código.
* **Acolhimento Local:** Prioridade total aos dados do arquivo `telefones_uteis_estado.pdf` no ranking de similaridade do FAISS, garantindo que informações críticas do RN apareçam primeiro.

---


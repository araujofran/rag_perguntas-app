import streamlit as st
import os
import json
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

# --- CONFIGURAÇÕES DO PROJETO ---
PASTA_PROJETO = os.path.dirname(os.path.abspath(__file__)) # Assume que o script está na pasta NOVORAG
PASTA_EMBEDDINGS = os.path.join(PASTA_PROJETO, "04_embeddings")

# --- ARQUIVOS DE DADOS RAG ---
ARQUIVO_EMBEDDINGS = os.path.join(PASTA_EMBEDDINGS, "embeddings_dados.npy")
ARQUIVO_METADADOS = os.path.join(PASTA_EMBEDDINGS, "metadados_limpos.json")

# --- PARÂMETROS ---
MODELO_EMBEDDING = 'all-MiniLM-L6-v2'
MODELO_LLM_RAG = 'daycoval-rag' # Nome que seria usado
MODELO_LLM_FALLBACK = 'mistral:latest' # Usado se o ajustado falhar ou não existir

# --- Variáveis de Sessão e Cache (Streamlit) ---

@st.cache_resource
def carregar_modelo_embedding():
    """Carrega e armazena o modelo de embedding na memória."""
    try:
        model = SentenceTransformer(MODELO_EMBEDDING)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de embedding: {e}")
        return None

@st.cache_resource
def carregar_dados_rag():
    """Carrega embeddings e textos de contexto do disco."""
    try:
        # 1. Carregar Embeddings
        embeddings = np.load(ARQUIVO_EMBEDDINGS)
        
        # 2. Carregar Metadados
        with open(ARQUIVO_METADADOS, 'r', encoding='utf-8') as f:
            metadados = json.load(f)
        
        # 3. Preparar Textos de Contexto
        # Formato: "Pergunta: ... | Resposta: ... | Fonte: ..."
        textos_contexto = [
            f"Pergunta: {item['pergunta_limpa']} | Resposta: {item['resposta_limpa_final']} | Fonte: {item['origem_base']}" 
            for item in metadados
        ]
        
        return embeddings, textos_contexto
    except Exception as e:
        st.error(f"Erro ao carregar dados RAG: {e}. Verifique os arquivos em 04_embeddings.")
        return None, None

def buscar_contexto_relevante(query_embedding_model, embeddings_base, textos_contexto, query, top_k=3):
    """Realiza a busca de similaridade vetorial (Retrieval)."""
    # 1. Vetoriza a pergunta do usuário
    query_embedding = query_embedding_model.encode(query, convert_to_tensor=False)
    
    # 2. Calcula a similaridade
    similarities = np.dot(query_embedding, embeddings_base.T)
    
    # 3. Encontra os índices dos top_k mais similares
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 4. Recupera os textos de contexto
    contexto_relevante = [textos_contexto[i] for i in top_k_indices]
    
    return "\n---\n".join(contexto_relevante)

def gerar_resposta_llm(contexto, query):
    """Gera a resposta usando Ollama (Generation)."""
    
    # System Prompt (simulando o modelo ajustado)
    system_prompt = (
        "Você é um assistente de Suporte Técnico Daycoval e especialista em produtos bancários e Pix. "
        "Sua função é usar o CONTEXTO fornecido para responder à pergunta do cliente de forma concisa, "
        "precisa e profissional. Sua resposta DEVE OBRIGATORIAMENTE terminar com a citação da FONTE(S) "
        "encontrada no CONTEXTO. Se o contexto for insuficiente, responda que a informação não está disponível."
    )
    
    # Prompt de Geração
    prompt_final = f"""
    Contexto de Suporte Técnico Daycoval:
    {contexto}

    ---
    
    Pergunta do Cliente: {query}
    
    Resposta:
    """
    
    try:
        # A API Ollama será usada. Verificamos qual modelo usar.
        client = ollama.Client()
        
        # O modelo daycoval-rag provavelmente não existe, então usamos o fallback
        modelo_final = MODELO_LLM_FALLBACK 
        
        response = client.generate(
            model=modelo_final,
            prompt=prompt_final,
            system=system_prompt,
            stream=False,
            options={"temperature": 0.0}
        )
        
        return response['response'].strip()
        
    except Exception as e:
        st.error(f"Erro de conexão com o Ollama. Verifique se o serviço está ativo (ícone na bandeja). Erro: {e}")
        return "Desculpe, não foi possível conectar ao modelo de geração de texto (LLM)."

# --- INTERFACE STREAMLIT ---

# 1. Carregar recursos na inicialização
embedding_model = carregar_modelo_embedding()
embeddings_base, textos_contexto = carregar_dados_rag()

st.set_page_config(page_title="Daycoval RAG Chatbot", layout="wide")

st.title("🏦 Suporte Técnico Daycoval (RAG/Ollama)")
st.caption("Arquitetura: Mistral (Simulado Fine-Tuning) + Embeddings S-BERT | Domínio: Produtos Bancários")

if embedding_model is None or embeddings_base is None:
    st.warning("O aplicativo não pode funcionar sem os modelos e dados carregados.")
else:
    # 2. Inicializar histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Mensagem inicial do assistente
        st.session_state.messages.append(
            {"role": "assistant", "content": "Olá! Sou o assistente de Suporte Daycoval. Como posso ajudar com suas dúvidas sobre Pix ou outros produtos?"}
        )

    # 3. Exibir histórico de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. Processar nova entrada do usuário
    if prompt := st.chat_input("Pergunte sobre Pix, câmbio ou financiamento..."):
        # Adiciona a pergunta do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Inicia a resposta do assistente (RAG)
        with st.chat_message("assistant"):
            with st.spinner("🤖 Consultando Base de Conhecimento e Gerando Resposta..."):
                try:
                    # 4.1. RECUPERAÇÃO (Retrieval)
                    contexto_relevante = buscar_contexto_relevante(
                        embedding_model, 
                        embeddings_base, 
                        textos_contexto, 
                        prompt, 
                        top_k=3
                    )
                    
                    # 4.2. GERAÇÃO (Generation)
                    resposta = gerar_resposta_llm(contexto_relevante, prompt)
                    
                    # 4.3. Exibe a resposta
                    st.markdown(resposta)
                    
                    # 4.4. Armazena a resposta no histórico
                    st.session_state.messages.append({"role": "assistant", "content": resposta})

                    # Opcional: Mostrar o contexto usado para fins de debugging/demonstração
                    with st.expander("Contexto (RAG) Utilizado"):
                        st.text(contexto_relevante)

                except Exception as e:
                    error_message = f"Ocorreu um erro no processo RAG: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
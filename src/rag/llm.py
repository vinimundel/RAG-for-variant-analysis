"""
src/rag/llm.py — Configuração do LLM local via Ollama.

Ollama é um servidor local de LLMs que:
    - Serve modelos em formato GGUF/llama.cpp com quantização eficiente
    - Expõe API REST compatível com OpenAI (/api/chat, /api/generate)
    - O LangChain se conecta via langchain-ollama sem autenticação

Modelo: llama3.1:8b
    - 8 bilhões de parâmetros (cabe em RAM/VRAM de consumidor)
    - Llama 3.1: melhor instruction following e raciocínio que versões anteriores
    - Contexto de 128K tokens (usaremos ~4096 para RAG)

Parâmetros críticos:
    temperature=0.1 → respostas determinísticas e factuais (não criativas)
    num_ctx=4096    → janela de contexto: 5 chunks × ~400 chars + pergunta + instrução
    num_predict=512 → limita o tamanho máximo da resposta (evita divagação)
"""

import os
from langchain_ollama import ChatOllama

# URL base do Ollama — sobrescrita pela variável de ambiente no Docker
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def llm_model_name() -> str:
    """Return the exact configured model identifier for output provenance."""
    return _MODEL_NAME


def get_llm(
    temperature: float = 0.1,
    num_ctx: int = 4096,
    num_predict: int = 512,
) -> ChatOllama:
    """
    Retorna o LLM Llama 3.1 8B via Ollama configurado para o RAG biomédico.

    Args:
        temperature:  0.1 = determinístico (ideal para Q&A factual)
                      0.7+ = criativo (não recomendado para literatura médica)
        num_ctx:      Janela de contexto em tokens. 4096 é suficiente para:
                      5 chunks × 512 tokens + instrução + resposta
        num_predict:  Máximo de tokens gerados na resposta.

    Returns:
        ChatOllama configurado e pronto para uso na chain

    Nota:
        ChatOllama usa a API /api/chat (mensagens) em vez de /api/generate (completion).
        Isso é importante porque o prompt template do RAG usa ChatPromptTemplate com
        HumanMessage/SystemMessage, que requerem a interface de chat.
    """
    return ChatOllama(
        model=_MODEL_NAME,
        base_url=_OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )

"""Configurable local Ollama model through LangChain."""
import os
from langchain_ollama import ChatOllama


def llm_model_name() -> str:
    return os.getenv('OLLAMA_MODEL', 'qwen2.5:1.5b')


def get_llm(temperature: float = 0.0, num_ctx: int = 4096, num_predict: int = 800):
    return ChatOllama(model=llm_model_name(),
                      base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                      temperature=temperature, num_ctx=num_ctx, num_predict=num_predict,
                      num_gpu=int(os.getenv('OLLAMA_NUM_GPU', '0')),
                      keep_alive=os.getenv('OLLAMA_KEEP_ALIVE', '0'),
                      client_kwargs={'timeout': float(os.getenv('OLLAMA_TIMEOUT', '180'))})

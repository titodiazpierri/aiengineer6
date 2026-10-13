import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE = BASE_DIR / "knowledge_base" / "documentos.json"
TRAZAS_DIR = BASE_DIR / "trazas"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
MODELOS_POR_DEFECTO: dict[str, str] = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5",
}
LLM_MODEL = os.getenv("LLM_MODEL") or MODELOS_POR_DEFECTO.get(LLM_PROVIDER, "")

MAX_DELEGACIONES = 6
MAX_REINTENTOS_VALIDACION = 1
RECURSION_LIMIT_GRAFO = 30
RECURSION_LIMIT_ESPECIALISTA = 12


def crear_llm(max_tokens: int = 1500, api_key: str | None = None) -> BaseChatModel:
    parametros: dict = {"model": LLM_MODEL, "temperature": 0, "max_tokens": max_tokens, "timeout": 60, "max_retries": 2}
    if api_key:
        parametros["api_key"] = api_key
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**parametros)
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**parametros)
    raise ValueError(f"LLM_PROVIDER inválido: {LLM_PROVIDER}. Usar openai o anthropic")


def verificar_credenciales() -> None:
    variable = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(LLM_PROVIDER)
    if variable and not os.getenv(variable):
        raise RuntimeError(f"Falta {variable}. Copiá .env.example a .env y completá la clave.")

import json
import re
import unicodedata
from functools import cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from agents.common import ejecutar_especialista
from config import KNOWLEDGE_BASE
from state import OrchestratorState

STOPWORDS = {
    "a", "al", "de", "del", "el", "en", "es", "la", "las", "lo", "los", "para", "por", "que",
    "se", "su", "sus", "un", "una", "y", "o", "con", "como", "cual", "cuales", "cuanto", "cuanta",
}

PROMPT_INVESTIGADOR = """Sos el Agente Investigador de un equipo de análisis. Tu único trabajo es encontrar datos en la base documental de la empresa usando tus herramientas. No hacés cálculos ni opinás.

Cómo trabajar:
1. Usá buscar_documentos con términos concretos. Si la primera búsqueda no alcanza, probá con otros términos.
2. Usá leer_documento para leer el contenido completo de cada documento relevante antes de reportar cifras.
3. Si encontrás cifras contradictorias entre documentos, reportá ambas con su doc_id, tipo y fecha, e indicá cuál es la fuente oficial más reciente.

Formato de tu respuesta final:
- DATOS: cada dato con su valor exacto tal como figura en el documento, y la fuente entre corchetes [doc_id, tipo, fecha].
- TEXTOS: si la instrucción pide reseñas u opiniones, transcribilas textualmente, una por línea.
- CONFLICTOS: cifras contradictorias encontradas, o "ninguno".
- FALTANTES: datos pedidos que no encontraste, o "ninguno"."""


def normalizar(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def terminos(texto: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalizar(texto)) if t not in STOPWORDS and len(t) > 1]


@cache
def cargar_documentos() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(KNOWLEDGE_BASE.read_text(encoding="utf-8")))


def texto_completo(documento: dict[str, Any]) -> str:
    return " ".join([documento["titulo"], documento["contenido"], *documento.get("resenas", [])])


@tool
def buscar_documentos(consulta: str, max_resultados: int = 3) -> list[dict[str, Any]]:
    """Busca documentos en la base documental de la empresa Nubeteca (informes trimestrales oficiales,
    notas de prensa, políticas internas y reseñas de clientes) por palabras clave.

    Usar para descubrir qué documentos contienen la información pedida. Devuelve un fragmento corto
    de cada uno; para leer cifras completas hay que usar leer_documento con el doc_id.

    Args:
        consulta: palabras clave en español, por ejemplo "ingresos segundo trimestre 2026" o "reseñas clientes".
        max_resultados: cantidad máxima de documentos a devolver, entre 1 y 5.

    Returns:
        Lista ordenada por relevancia con doc_id, titulo, tipo, fecha, puntaje y fragmento.
        Lista vacía si ningún documento coincide.
    """
    buscados = set(terminos(consulta))
    resultados = []
    for documento in cargar_documentos():
        palabras = terminos(texto_completo(documento))
        titulo = set(terminos(documento["titulo"]))
        puntaje = sum(palabras.count(t) for t in buscados) + 3 * len(buscados & titulo)
        if puntaje > 0:
            resultados.append({
                "doc_id": documento["doc_id"],
                "titulo": documento["titulo"],
                "tipo": documento["tipo"],
                "fecha": documento["fecha"],
                "puntaje": puntaje,
                "fragmento": documento["contenido"][:180] + "...",
            })
    resultados.sort(key=lambda r: r["puntaje"], reverse=True)
    return resultados[: max(1, min(max_resultados, 5))]


@tool
def leer_documento(doc_id: str) -> dict[str, Any]:
    """Devuelve el contenido completo de un documento de la base documental, incluidas las reseñas
    textuales si el documento las tiene.

    Usar después de buscar_documentos para obtener las cifras exactas, la fecha y el tipo de fuente.

    Args:
        doc_id: identificador exacto devuelto por buscar_documentos, por ejemplo "informe-q2-2026".

    Returns:
        doc_id, titulo, tipo, fecha, contenido y, si corresponde, la lista de reseñas.
        Si el doc_id no existe, un campo "error" con los doc_id disponibles.
    """
    for documento in cargar_documentos():
        if documento["doc_id"] == doc_id.strip():
            return dict(documento)
    return {"error": f"No existe el documento '{doc_id}'.", "doc_ids_disponibles": [d["doc_id"] for d in cargar_documentos()]}


HERRAMIENTAS_INVESTIGADOR = [buscar_documentos, leer_documento]


def crear_investigador(llm: BaseChatModel) -> CompiledStateGraph:
    return create_react_agent(llm, HERRAMIENTAS_INVESTIGADOR, prompt=PROMPT_INVESTIGADOR, name="investigador")


def crear_nodo_investigador(llm: BaseChatModel):
    agente = crear_investigador(llm)

    async def nodo_investigador(state: OrchestratorState) -> dict[str, Any]:
        contexto = ""
        if state.get("feedback_validacion"):
            contexto = f"Observaciones de la validación anterior: {state['feedback_validacion']}"
        return await ejecutar_especialista(
            agente, "investigador", state["instruccion_actual"], contexto, state["delegaciones"] + 1
        )

    return nodo_investigador

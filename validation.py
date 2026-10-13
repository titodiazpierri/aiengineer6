import logging
import re
from typing import Any, Literal

from langgraph.graph import END

from config import MAX_REINTENTOS_VALIDACION
from state import OrchestratorState, ResultadoValidacion, contribuciones_de

logger = logging.getLogger("validador")

PATRON_NUMERO = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*(?![\w])")


def a_numero(token: str) -> float | None:
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", token):
        token = token.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", token):
        token = token.replace(",", "")
    else:
        token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def extraer_numeros(texto: str) -> set[float]:
    numeros = (a_numero(t) for t in PATRON_NUMERO.findall(texto))
    return {n for n in numeros if n is not None}


def es_relevante(numero: float) -> bool:
    es_anio = numero.is_integer() and 2000 <= numero <= 2100
    return abs(numero) >= 10 and not es_anio


def respaldado(numero: float, evidencia: set[float]) -> bool:
    return any(abs(numero - e) <= max(0.051, 0.005 * abs(e)) for e in evidencia)


def corpus_de_evidencia(state: OrchestratorState) -> str:
    partes = [state["consulta"]]
    for aporte in state.get("contribuciones", []):
        partes.append(aporte["salida"])
        partes.extend(aporte["evidencia"])
    return "\n".join(partes)


def validar_resultado(state: OrchestratorState) -> ResultadoValidacion:
    problemas: list[str] = []
    investigacion = contribuciones_de(state, "investigador")
    analisis = contribuciones_de(state, "analista")

    if not any(a["herramientas"] for a in investigacion):
        problemas.append("Ningún aporte del investigador usó herramientas de búsqueda: los datos no están respaldados por documentos.")
    if not any(a["herramientas"] for a in analisis):
        problemas.append("Ningún aporte del analista usó herramientas de cómputo: los cálculos no son verificables.")

    respuesta = state.get("respuesta_final", "").strip()
    if not respuesta:
        problemas.append("La respuesta final está vacía.")
    else:
        evidencia = extraer_numeros(corpus_de_evidencia(state))
        sin_respaldo = sorted(n for n in extraer_numeros(respuesta) if es_relevante(n) and not respaldado(n, evidencia))
        if sin_respaldo:
            valores = ", ".join(f"{n:g}" for n in sin_respaldo)
            problemas.append(
                f"La respuesta final incluye cifras que no aparecen en ningún aporte ni resultado de herramienta: {valores}. "
                "Deben calcularse con el analista o eliminarse."
            )
        if not re.search(r"\[[a-z0-9-]+\]", respuesta):
            problemas.append("La respuesta final no cita ninguna fuente con su doc_id entre corchetes.")

    return {"aprobada": not problemas, "problemas": problemas, "intento": len(state.get("validaciones", [])) + 1}


async def nodo_validador(state: OrchestratorState) -> dict[str, Any]:
    resultado = validar_resultado(state)
    if resultado["aprobada"]:
        logger.info("Validación aprobada en el intento %d", resultado["intento"])
        return {"validaciones": [resultado], "task_completed": True, "feedback_validacion": ""}

    logger.warning("Validación rechazada (intento %d): %s", resultado["intento"], " | ".join(resultado["problemas"]))
    agota_reintentos = resultado["intento"] > MAX_REINTENTOS_VALIDACION
    actualizacion: dict[str, Any] = {
        "validaciones": [resultado],
        "task_completed": False,
        "feedback_validacion": " ".join(resultado["problemas"]),
    }
    if agota_reintentos:
        aviso = "\n\nAdvertencia de validación: " + " ".join(resultado["problemas"])
        actualizacion["respuesta_final"] = state.get("respuesta_final", "") + aviso
    return actualizacion


def enrutar_validacion(state: OrchestratorState) -> Literal["supervisor", "__end__"]:
    ultima = state["validaciones"][-1]
    if ultima["aprobada"] or ultima["intento"] > MAX_REINTENTOS_VALIDACION:
        return END
    return "supervisor"

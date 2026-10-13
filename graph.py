from pathlib import Path
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst_agent import crear_nodo_analista
from agents.research_agent import crear_nodo_investigador
from agents.supervisor import crear_nodo_supervisor
from agents.synthesizer_agent import crear_nodo_sintetizador
from config import crear_llm
from state import OrchestratorState
from validation import enrutar_validacion, nodo_validador


def enrutar_supervisor(state: OrchestratorState) -> Literal["investigador", "analista", "sintetizador"]:
    return state["next_agent"]


def construir_grafo(llm: BaseChatModel | None = None) -> CompiledStateGraph:
    llm = llm or crear_llm()

    builder = StateGraph(OrchestratorState)
    builder.add_node("supervisor", crear_nodo_supervisor(llm))
    builder.add_node("investigador", crear_nodo_investigador(llm))
    builder.add_node("analista", crear_nodo_analista(llm))
    builder.add_node("sintetizador", crear_nodo_sintetizador(llm))
    builder.add_node("validador", nodo_validador)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", enrutar_supervisor)
    builder.add_edge("investigador", "supervisor")
    builder.add_edge("analista", "supervisor")
    builder.add_edge("sintetizador", "validador")
    builder.add_conditional_edges("validador", enrutar_validacion)
    return builder.compile()


def exportar_diagrama(grafo: CompiledStateGraph, directorio: Path) -> list[Path]:
    directorio.mkdir(parents=True, exist_ok=True)
    estructura = grafo.get_graph()
    ruta_mermaid = directorio / "grafo.mmd"
    ruta_mermaid.write_text(estructura.draw_mermaid(), encoding="utf-8")
    generados = [ruta_mermaid]
    try:
        ruta_png = directorio / "grafo.png"
        ruta_png.write_bytes(estructura.draw_mermaid_png())
        generados.append(ruta_png)
    except Exception as error:
        print(f"No se pudo generar grafo.png ({type(error).__name__}). El diagrama Mermaid quedó en {ruta_mermaid.name}.")
    return generados

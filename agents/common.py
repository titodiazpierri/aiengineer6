from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from config import RECURSION_LIMIT_ESPECIALISTA
from state import AgenteEspecialista, Contribucion

MAX_CARACTERES_EVIDENCIA = 3000


async def ejecutar_especialista(
    agente: CompiledStateGraph,
    nombre: AgenteEspecialista,
    instruccion: str,
    contexto: str,
    iteracion: int,
) -> dict[str, Any]:
    mensaje = f"INSTRUCCIÓN DEL SUPERVISOR:\n{instruccion}"
    if contexto:
        mensaje += f"\n\nCONTEXTO DISPONIBLE:\n{contexto}"

    resultado = await agente.ainvoke(
        {"messages": [HumanMessage(mensaje)]},
        {"recursion_limit": RECURSION_LIMIT_ESPECIALISTA},
    )
    mensajes = resultado["messages"]

    herramientas = [llamada["name"] for m in mensajes if isinstance(m, AIMessage) for llamada in m.tool_calls]
    evidencia = [f"[{m.name}] {str(m.content)[:MAX_CARACTERES_EVIDENCIA]}" for m in mensajes if isinstance(m, ToolMessage)]
    finales = [m for m in mensajes if isinstance(m, AIMessage) and not m.tool_calls]
    salida = finales[-1].text if finales else "El especialista no produjo una respuesta final."

    contribucion: Contribucion = {
        "agente": nombre,
        "iteracion": iteracion,
        "instruccion": instruccion,
        "salida": salida,
        "herramientas": herramientas,
        "evidencia": evidencia,
    }
    return {
        "contribuciones": [contribucion],
        "delegaciones": iteracion,
        "messages": [AIMessage(content=salida, name=nombre)],
    }

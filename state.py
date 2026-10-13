import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

AgenteEspecialista = Literal["investigador", "analista"]
Destino = Literal["investigador", "analista", "sintetizador"]


class Contribucion(TypedDict):
    agente: AgenteEspecialista
    iteracion: int
    instruccion: str
    salida: str
    herramientas: list[str]
    evidencia: list[str]


class Decision(TypedDict):
    iteracion: int
    siguiente: Destino
    instruccion: str
    razonamiento: str
    evaluacion_ultimo_aporte: str
    forzada: bool


class ResultadoValidacion(TypedDict):
    aprobada: bool
    problemas: list[str]
    intento: int


class OrchestratorState(MessagesState):
    consulta: str
    next_agent: Destino
    instruccion_actual: str
    delegaciones: int
    contribuciones: Annotated[list[Contribucion], operator.add]
    decisiones: Annotated[list[Decision], operator.add]
    feedback_validacion: str
    validaciones: Annotated[list[ResultadoValidacion], operator.add]
    respuesta_final: str
    task_completed: bool


def estado_inicial(consulta: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(consulta)],
        "consulta": consulta,
        "next_agent": "investigador",
        "instruccion_actual": "",
        "delegaciones": 0,
        "contribuciones": [],
        "decisiones": [],
        "feedback_validacion": "",
        "validaciones": [],
        "respuesta_final": "",
        "task_completed": False,
    }


def contribuciones_de(state: OrchestratorState, agente: AgenteEspecialista) -> list[Contribucion]:
    return [c for c in state.get("contribuciones", []) if c["agente"] == agente]

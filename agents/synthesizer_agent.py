from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from state import OrchestratorState

PROMPT_SINTETIZADOR = """Sos el Agente Sintetizador. Redactás la respuesta final para el usuario a partir de los aportes de los especialistas.

Reglas estrictas:
1. Usá únicamente información presente en los APORTES. No agregues datos ni hagas cálculos nuevos.
2. Copiá cada cifra exactamente como la calculó el analista o la reportó el investigador.
3. Citá la fuente de los datos entre corchetes con su doc_id, por ejemplo [informe-q2-2026].
4. Si hubo cifras contradictorias, aclará cuál se usó y por qué se descartó la otra.
5. Estructura: un párrafo de respuesta directa, luego "Detalle" con viñetas, y una "Conclusión" de una o dos oraciones.
6. Respondé en español."""

PROMPT_DATOS = """CONSULTA DEL USUARIO:
{consulta}

APORTES:
{aportes}

{observaciones}"""

prompt_sintetizador = ChatPromptTemplate.from_messages([("system", PROMPT_SINTETIZADOR), ("human", PROMPT_DATOS)])


def formatear_para_sintesis(state: OrchestratorState) -> str:
    return "\n\n".join(
        f"[{aporte['agente']} - iteración {aporte['iteracion']}]\n{aporte['salida']}"
        for aporte in state.get("contribuciones", [])
    )


def crear_nodo_sintetizador(llm: BaseChatModel):
    cadena = prompt_sintetizador | llm | StrOutputParser()

    async def nodo_sintetizador(state: OrchestratorState) -> dict[str, Any]:
        observaciones = ""
        if state.get("feedback_validacion"):
            observaciones = f"CORRECCIONES PEDIDAS POR LA VALIDACIÓN ANTERIOR (resolvelas):\n{state['feedback_validacion']}"
        respuesta = await cadena.ainvoke({
            "consulta": state["consulta"],
            "aportes": formatear_para_sintesis(state),
            "observaciones": observaciones,
        })
        return {"respuesta_final": respuesta, "messages": [AIMessage(content=respuesta, name="sintetizador")]}

    return nodo_sintetizador

import logging
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import MAX_DELEGACIONES
from state import Decision, Destino, OrchestratorState, contribuciones_de

logger = logging.getLogger("supervisor")

MAX_CARACTERES_APORTE = 2500

PROMPT_SUPERVISOR = """Sos el Supervisor de un equipo multi-agente de análisis e investigación. No resolvés la tarea vos: decidís quién interviene ahora o si es momento de finalizar.

Especialistas disponibles:
- investigador: busca y lee documentos de la base documental (informes oficiales, notas de prensa, políticas, reseñas). No calcula.
- analista: hace cálculos (variaciones porcentuales, estadísticas, expresiones) y análisis de sentimiento con herramientas, sobre los datos que ya aportó el investigador. No busca información.

Opciones de "siguiente":
- "investigador" o "analista": delegar con una instrucción concreta y autocontenida. El especialista NO ve la conversación: escribí en la instrucción todo lo que necesita saber.
- "finalizar": pasar a la síntesis final. Solo si se cumple TODA la rúbrica.

RÚBRICA DE VALIDACIÓN (evaluá el último aporte y el conjunto antes de finalizar):
1. Cobertura: cada parte de la consulta del usuario tiene datos y, si corresponde, un cálculo.
2. Trazabilidad: cada cifra reportada por el investigador indica su documento de origen (doc_id y fecha).
3. Conflictos resueltos: si hay cifras contradictorias, se identificó la fuente oficial más reciente y se usa esa.
4. Cómputo verificable: todo porcentaje, diferencia, promedio o sentimiento lo calculó el analista con una herramienta.
5. Consistencia: el analista usó exactamente las cifras aportadas por el investigador.

Reglas de delegación:
- Si faltan datos, delegá al investigador. Si los datos están pero falta procesarlos, delegá al analista.
- Si un aporte no cumple la rúbrica, re-delegá al mismo especialista indicando exactamente qué corregir.
- No repitas una instrucción idéntica a una anterior que ya se cumplió.
- Delegaciones usadas: {delegaciones} de {maximo}. Si quedan pocas, priorizá lo imprescindible."""

PROMPT_ESTADO = """CONSULTA DEL USUARIO:
{consulta}

APORTES DE LOS ESPECIALISTAS:
{aportes}

OBSERVACIONES DE LA VALIDACIÓN AUTOMÁTICA:
{feedback}

Decidí el siguiente paso."""


class DecisionSupervisor(BaseModel):
    evaluacion_ultimo_aporte: str = Field(
        description="Evaluación del último aporte según la rúbrica: qué criterios cumple y cuáles no. 'Sin aportes todavía' si no hay ninguno."
    )
    razonamiento: str = Field(description="Por qué se elige el siguiente paso, en una o dos oraciones.")
    siguiente: Literal["investigador", "analista", "finalizar"] = Field(
        description="Quién interviene ahora, o 'finalizar' si se cumple toda la rúbrica."
    )
    instruccion: str = Field(
        description="Instrucción concreta y autocontenida para el especialista elegido. Vacía si siguiente es 'finalizar'."
    )


prompt_supervisor = ChatPromptTemplate.from_messages([("system", PROMPT_SUPERVISOR), ("human", PROMPT_ESTADO)])


def formatear_aportes(state: OrchestratorState) -> str:
    aportes = state.get("contribuciones", [])
    if not aportes:
        return "Ninguno todavía."
    bloques = []
    for aporte in aportes:
        herramientas = ", ".join(aporte["herramientas"]) or "ninguna"
        salida = aporte["salida"][:MAX_CARACTERES_APORTE]
        bloques.append(
            f"[Iteración {aporte['iteracion']} - {aporte['agente']}] Instrucción: {aporte['instruccion']}\n"
            f"Herramientas usadas: {herramientas}\nResultado:\n{salida}"
        )
    return "\n\n".join(bloques)


def aplicar_criterio_de_suficiencia(state: OrchestratorState, decision: DecisionSupervisor) -> tuple[Destino, str, bool]:
    tiene_investigacion = bool(contribuciones_de(state, "investigador"))
    tiene_analisis = bool(contribuciones_de(state, "analista"))
    if decision.siguiente == "analista" and not tiene_investigacion:
        return "investigador", f"Antes del análisis hacen falta datos. Buscá la información necesaria para responder: {state['consulta']}", True
    if decision.siguiente == "finalizar" and not tiene_investigacion:
        return "investigador", f"Buscá en la base documental la información necesaria para responder: {state['consulta']}", True
    if decision.siguiente == "finalizar" and not tiene_analisis:
        return "analista", f"Procesá con tus herramientas los datos aportados por el investigador para responder: {state['consulta']}", True
    if decision.siguiente == "finalizar":
        return "sintetizador", "", False
    return decision.siguiente, decision.instruccion, False


def crear_nodo_supervisor(llm: BaseChatModel):
    cadena = prompt_supervisor | llm.with_structured_output(DecisionSupervisor, method="function_calling")

    async def nodo_supervisor(state: OrchestratorState) -> dict[str, Any]:
        delegaciones = state.get("delegaciones", 0)
        iteracion = delegaciones + 1

        if delegaciones >= MAX_DELEGACIONES:
            decision_registrada: Decision = {
                "iteracion": iteracion,
                "siguiente": "sintetizador",
                "instruccion": "",
                "razonamiento": f"Se alcanzó el máximo de {MAX_DELEGACIONES} delegaciones: se finaliza con la información disponible.",
                "evaluacion_ultimo_aporte": "No evaluado por límite de pasos.",
                "forzada": True,
            }
        else:
            decision: DecisionSupervisor = await cadena.ainvoke({
                "consulta": state["consulta"],
                "aportes": formatear_aportes(state),
                "feedback": state.get("feedback_validacion") or "Ninguna.",
                "delegaciones": delegaciones,
                "maximo": MAX_DELEGACIONES,
            })
            siguiente, instruccion, forzada = aplicar_criterio_de_suficiencia(state, decision)
            razonamiento = decision.razonamiento
            if forzada:
                razonamiento += f" [Criterio de suficiencia: se redirige a '{siguiente}' porque falta un aporte obligatorio.]"
            decision_registrada = {
                "iteracion": iteracion,
                "siguiente": siguiente,
                "instruccion": instruccion,
                "razonamiento": razonamiento,
                "evaluacion_ultimo_aporte": decision.evaluacion_ultimo_aporte,
                "forzada": forzada,
            }

        logger.info("Supervisor → %s | %s", decision_registrada["siguiente"], decision_registrada["razonamiento"])
        destino = decision_registrada["siguiente"]
        texto = f"Delego a {destino}: {decision_registrada['instruccion']}" if destino != "sintetizador" else "La tarea cumple la rúbrica: paso a la síntesis final."
        return {
            "next_agent": destino,
            "instruccion_actual": decision_registrada["instruccion"],
            "decisiones": [decision_registrada],
            "messages": [AIMessage(content=texto, name="supervisor")],
        }

    return nodo_supervisor

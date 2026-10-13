import ast
import operator
import re
import statistics
import unicodedata
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from agents.common import ejecutar_especialista
from state import OrchestratorState, contribuciones_de

MAX_CARACTERES_CONTEXTO = 7000

PROMPT_ANALISTA = """Sos el Agente Analista de un equipo de análisis. Tu trabajo es procesar los datos que aportó el Agente Investigador usando tus herramientas de cómputo. No buscás información nueva.

Reglas:
1. Todo cálculo (porcentajes, diferencias, promedios, sentimiento) se hace con una herramienta. Nunca hagas cuentas de memoria.
2. Usá exactamente las cifras y textos que figuran en el CONTEXTO DISPONIBLE. Si hay cifras contradictorias, usá la que el investigador marcó como fuente oficial más reciente.
3. Si te faltan datos para cumplir la instrucción, no los inventes: indicalo en FALTANTES.

Formato de tu respuesta final:
- RESULTADOS: cada resultado con su valor exacto devuelto por la herramienta y qué herramienta lo calculó.
- DATOS USADOS: las cifras de entrada y su fuente.
- FALTANTES: datos que necesitarías y no tenés, o "ninguno"."""

OPERADORES: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

PREFIJOS_POSITIVOS = ("excelent", "intuitiv", "rapid", "recomiend", "recomend", "facil", "util", "ahorr", "resolv", "encant", "estable", "mejor", "buen")
PREFIJOS_NEGATIVOS = ("caid", "perd", "problem", "lent", "error", "confus", "falt", "fall", "dificil", "tard", "incident", "pesim")
EXACTAS_POSITIVAS = {"bien"}
EXACTAS_NEGATIVAS = {"caro", "cara", "caros", "caras", "mal", "mala", "malo", "malos", "malas"}
NEGACIONES = {"no", "nunca", "ni", "tampoco"}


def _evaluar_nodo(nodo: ast.AST) -> float:
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)):
        return float(nodo.value)
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in OPERADORES:
        izquierda, derecha = _evaluar_nodo(nodo.left), _evaluar_nodo(nodo.right)
        if isinstance(nodo.op, ast.Pow) and abs(derecha) > 10:
            raise ValueError("Exponente demasiado grande")
        return OPERADORES[type(nodo.op)](izquierda, derecha)
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in OPERADORES:
        return OPERADORES[type(nodo.op)](_evaluar_nodo(nodo.operand))
    raise ValueError("Solo se permiten números y los operadores + - * / ** %")


@tool
def evaluar_expresion(expresion: str) -> dict[str, Any]:
    """Evalúa una expresión aritmética de forma segura y devuelve el resultado redondeado a 2 decimales.

    Usar para cualquier cuenta que no cubran las otras herramientas, por ejemplo un ingreso por cliente.

    Args:
        expresion: expresión con números sin separadores de miles y punto decimal, por ejemplo "2116000 / 341".

    Returns:
        expresion y resultado. Si la expresión no es válida, un campo "error".
    """
    try:
        resultado = _evaluar_nodo(ast.parse(expresion, mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError) as error:
        return {"error": f"Expresión inválida: {error}"}
    return {"expresion": expresion, "resultado": round(resultado, 2)}


@tool
def calcular_variacion_porcentual(valor_inicial: float, valor_final: float) -> dict[str, Any]:
    """Calcula la variación porcentual y la diferencia absoluta entre dos valores, por ejemplo el
    crecimiento de ingresos entre dos trimestres.

    Args:
        valor_inicial: valor del período anterior, sin separadores de miles. Ejemplo: 1840000.
        valor_final: valor del período posterior, sin separadores de miles. Ejemplo: 2116000.

    Returns:
        valor_inicial, valor_final, diferencia_absoluta y variacion_porcentual (redondeada a 2 decimales).
    """
    if valor_inicial == 0:
        return {"error": "El valor inicial no puede ser 0 para calcular una variación porcentual."}
    return {
        "valor_inicial": valor_inicial,
        "valor_final": valor_final,
        "diferencia_absoluta": round(valor_final - valor_inicial, 2),
        "variacion_porcentual": round((valor_final - valor_inicial) / valor_inicial * 100, 2),
    }


@tool
def calcular_estadisticas(valores: list[float]) -> dict[str, Any]:
    """Calcula estadísticas descriptivas de una lista de números: cantidad, suma, promedio, mediana,
    mínimo y máximo.

    Args:
        valores: lista de números sin separadores de miles, por ejemplo [312, 341].

    Returns:
        cantidad, suma, promedio, mediana, minimo y maximo, redondeados a 2 decimales.
    """
    if not valores:
        return {"error": "La lista de valores está vacía."}
    return {
        "cantidad": len(valores),
        "suma": round(sum(valores), 2),
        "promedio": round(statistics.mean(valores), 2),
        "mediana": round(statistics.median(valores), 2),
        "minimo": min(valores),
        "maximo": max(valores),
    }


def _normalizar(texto: str) -> list[str]:
    descompuesto = unicodedata.normalize("NFKD", texto.lower())
    return re.findall(r"[a-z]+", "".join(c for c in descompuesto if not unicodedata.combining(c)))


def _polaridad(palabra: str) -> int:
    if palabra in EXACTAS_POSITIVAS or palabra.startswith(PREFIJOS_POSITIVOS):
        return 1
    if palabra in EXACTAS_NEGATIVAS or palabra.startswith(PREFIJOS_NEGATIVOS):
        return -1
    return 0


def puntuar_texto(texto: str) -> dict[str, Any]:
    palabras = _normalizar(texto)
    positivas, negativas = [], []
    for indice, palabra in enumerate(palabras):
        polaridad = _polaridad(palabra)
        if polaridad and NEGACIONES & set(palabras[max(0, indice - 2):indice]):
            polaridad = -polaridad
        if polaridad > 0:
            positivas.append(palabra)
        elif polaridad < 0:
            negativas.append(palabra)
    total = len(positivas) + len(negativas)
    puntaje = round((len(positivas) - len(negativas)) / total, 2) if total else 0.0
    sentimiento = "positivo" if puntaje > 0.2 else "negativo" if puntaje < -0.2 else "neutral"
    return {"texto": texto, "puntaje": puntaje, "sentimiento": sentimiento, "positivas": positivas, "negativas": negativas}


@tool
def analizar_sentimiento(textos: list[str]) -> dict[str, Any]:
    """Analiza el sentimiento de una lista de textos en español (reseñas, comentarios u opiniones) con un
    léxico de palabras positivas y negativas que contempla negaciones como "no es mala".

    Usar cuando la instrucción pide el sentimiento, la satisfacción o la percepción de los clientes.
    Pasar cada reseña como un elemento separado de la lista, con el texto textual.

    Args:
        textos: lista de textos a analizar, uno por reseña.

    Returns:
        detalle por texto (puntaje entre -1 y 1, sentimiento, palabras detectadas) y un resumen con la
        cantidad de positivos, negativos y neutrales, el porcentaje de positivos, el puntaje promedio y
        el sentimiento general.
    """
    if not textos:
        return {"error": "No se recibieron textos para analizar."}
    detalle = [puntuar_texto(t) for t in textos]
    conteo = {s: sum(1 for d in detalle if d["sentimiento"] == s) for s in ("positivo", "negativo", "neutral")}
    promedio = round(statistics.mean(d["puntaje"] for d in detalle), 2)
    general = "positivo" if promedio > 0.2 else "negativo" if promedio < -0.2 else "mixto"
    return {
        "detalle": detalle,
        "resumen": {
            "total_textos": len(detalle),
            "positivos": conteo["positivo"],
            "negativos": conteo["negativo"],
            "neutrales": conteo["neutral"],
            "porcentaje_positivos": round(conteo["positivo"] / len(detalle) * 100, 2),
            "puntaje_promedio": promedio,
            "sentimiento_general": general,
        },
    }


HERRAMIENTAS_ANALISTA = [calcular_variacion_porcentual, calcular_estadisticas, analizar_sentimiento, evaluar_expresion]


def crear_analista(llm: BaseChatModel) -> CompiledStateGraph:
    return create_react_agent(llm, HERRAMIENTAS_ANALISTA, prompt=PROMPT_ANALISTA, name="analista")


def contexto_para_analista(state: OrchestratorState) -> str:
    partes = []
    for aporte in contribuciones_de(state, "investigador"):
        partes.append(f"--- Aporte del investigador (iteración {aporte['iteracion']}) ---\n{aporte['salida']}")
        documentos = [e for e in aporte["evidencia"] if e.startswith("[leer_documento]")]
        if documentos:
            partes.append("Documentos leídos por el investigador:\n" + "\n".join(documentos))
    if state.get("feedback_validacion"):
        partes.append(f"Observaciones de la validación anterior: {state['feedback_validacion']}")
    return "\n\n".join(partes)[-MAX_CARACTERES_CONTEXTO:]


def crear_nodo_analista(llm: BaseChatModel):
    agente = crear_analista(llm)

    async def nodo_analista(state: OrchestratorState) -> dict[str, Any]:
        return await ejecutar_especialista(
            agente, "analista", state["instruccion_actual"], contexto_para_analista(state), state["delegaciones"] + 1
        )

    return nodo_analista

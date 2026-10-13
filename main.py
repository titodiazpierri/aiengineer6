import argparse
import asyncio
import json
import logging
from collections import Counter
from datetime import datetime
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from config import (
    BASE_DIR,
    LLM_MODEL,
    LLM_PROVIDER,
    MAX_DELEGACIONES,
    RECURSION_LIMIT_GRAFO,
    TRAZAS_DIR,
    crear_llm,
    verificar_credenciales,
)
from graph import construir_grafo, exportar_diagrama
from state import estado_inicial

CONSULTA_DEMO = (
    "Analizá el desempeño de Nubeteca en 2026: ¿cuánto crecieron los ingresos entre el primer y el segundo "
    "trimestre, en porcentaje y en dólares, y cuál es el sentimiento general de las reseñas de clientes del "
    "segundo trimestre? Cerrá con una conclusión."
)

MAX_CARACTERES_PANTALLA = 700


class RegistroFlujo:
    def __init__(self, mostrar: bool = True) -> None:
        self.mostrar = mostrar
        self.lineas: list[str] = []
        self.eventos: list[dict[str, Any]] = []

    def linea(self, texto: str = "") -> None:
        self.lineas.append(texto)
        if self.mostrar:
            print(texto)

    def encabezado(self, paso: int, nodo: str) -> None:
        self.linea()
        self.linea(f"──── Paso {paso} · {nodo.upper()} " + "─" * max(0, 60 - len(nodo)))

    def registrar(self, paso: int, nodo: str, cambios: dict[str, Any]) -> None:
        self.encabezado(paso, nodo)
        evento: dict[str, Any] = {"paso": paso, "nodo": nodo}

        if nodo == "supervisor":
            decision = cambios["decisiones"][-1]
            self.linea(f"Evaluación del último aporte: {decision['evaluacion_ultimo_aporte']}")
            self.linea(f"Razonamiento: {decision['razonamiento']}")
            self.linea(f"Decisión: → {decision['siguiente']}" + (" (forzada)" if decision["forzada"] else ""))
            if decision["instruccion"]:
                self.linea(f"Instrucción: {decision['instruccion']}")
            evento["decision"] = decision

        elif nodo in ("investigador", "analista"):
            aporte = cambios["contribuciones"][-1]
            usos = Counter(aporte["herramientas"])
            self.linea("Herramientas usadas: " + (", ".join(f"{h} x{n}" for h, n in usos.items()) or "ninguna"))
            salida = aporte["salida"]
            recorte = salida if len(salida) <= MAX_CARACTERES_PANTALLA else salida[:MAX_CARACTERES_PANTALLA] + " [...]"
            self.linea(f"Resultado:\n{recorte}")
            evento["contribucion"] = aporte

        elif nodo == "sintetizador":
            self.linea("Respuesta final redactada a partir de los aportes.")
            evento["respuesta"] = cambios["respuesta_final"]

        elif nodo == "validador":
            resultado = cambios["validaciones"][-1]
            if resultado["aprobada"]:
                self.linea(f"✅ Validación aprobada (intento {resultado['intento']}).")
            else:
                self.linea(f"❌ Validación rechazada (intento {resultado['intento']}):")
                for problema in resultado["problemas"]:
                    self.linea(f"   - {problema}")
            evento["validacion"] = resultado

        self.eventos.append(evento)


async def ejecutar(consulta: str, grafo: CompiledStateGraph | None = None, mostrar: bool = True) -> tuple[dict[str, Any], RegistroFlujo]:
    grafo = grafo or construir_grafo()
    registro = RegistroFlujo(mostrar)
    registro.linea(f"CONSULTA: {consulta}")

    estado_final: dict[str, Any] = {}
    paso = 0
    async for modo, datos in grafo.astream(
        estado_inicial(consulta),
        {"recursion_limit": RECURSION_LIMIT_GRAFO},
        stream_mode=["updates", "values"],
    ):
        if modo == "values":
            estado_final = datos
            continue
        for nodo, cambios in datos.items():
            paso += 1
            registro.registrar(paso, nodo, cambios or {})

    registro.linea()
    registro.linea("=" * 66)
    registro.linea("RESPUESTA FINAL")
    registro.linea("=" * 66)
    registro.linea(estado_final.get("respuesta_final", ""))
    registro.linea()
    recorrido = " → ".join(evento["nodo"] for evento in registro.eventos)
    registro.linea(f"Recorrido de delegación: {recorrido}")
    registro.linea(
        f"Delegaciones a especialistas: {estado_final.get('delegaciones', 0)} de {MAX_DELEGACIONES} | "
        f"Tarea completada y validada: {'sí' if estado_final.get('task_completed') else 'no'}"
    )
    return estado_final, registro


def guardar_traza(consulta: str, estado: dict[str, Any], registro: RegistroFlujo, nombre: str = "ejecucion_demo") -> None:
    TRAZAS_DIR.mkdir(exist_ok=True)
    traza = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "proveedor": LLM_PROVIDER,
        "modelo": LLM_MODEL,
        "consulta": consulta,
        "flujo": registro.eventos,
        "contribuciones": estado.get("contribuciones", []),
        "decisiones_supervisor": estado.get("decisiones", []),
        "validaciones": estado.get("validaciones", []),
        "respuesta_final": estado.get("respuesta_final", ""),
        "task_completed": estado.get("task_completed", False),
    }
    (TRAZAS_DIR / f"{nombre}.json").write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    (TRAZAS_DIR / f"{nombre}.log").write_text("\n".join(registro.lineas) + "\n", encoding="utf-8")
    print(f"\nTraza guardada en {TRAZAS_DIR.name}/{nombre}.json y {TRAZAS_DIR.name}/{nombre}.log")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestador multi-agente con supervisor")
    parser.add_argument("consulta", nargs="?", default=CONSULTA_DEMO, help="Consulta a resolver")
    parser.add_argument("--diagrama", action="store_true", help="Solo exporta el diagrama del grafo a docs/")
    argumentos = parser.parse_args()

    if argumentos.diagrama:
        grafo = construir_grafo(crear_llm(api_key="solo-para-diagrama"))
        for ruta in exportar_diagrama(grafo, BASE_DIR / "docs"):
            print(f"Diagrama exportado: {ruta.relative_to(BASE_DIR)}")
        return

    verificar_credenciales()
    estado, registro = await ejecutar(argumentos.consulta)
    guardar_traza(argumentos.consulta, estado, registro)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    asyncio.run(main())

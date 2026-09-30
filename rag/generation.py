"""
Generación OPCIONAL: toma los
top-5 fragmentos que devuelve RAGSystem y le pide a un LLM una respuesta
basada SOLO en ellos. Se usa desde main.py con --responder.

El proveedor se elige por variables de entorno: LLM_PROVIDER si está
definida; si no, el primero que tenga API key, en este orden: OpenAI,
Anthropic, Gemini.
"""
import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from . import config  # noqa: F401  (importarlo carga el .env)
from .schemas import FuenteCitada, RespuestaRAG

PROVEEDORES = {
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-flash-latest"),
}

SYSTEM_PROMPT = """Sos un asistente que responde preguntas sobre modelos de Volkswagen
basándote ÚNICAMENTE en el CONTEXTO que se te proporciona.

Reglas:
1. Usá solo información presente en el CONTEXTO. No completes con conocimiento general.
2. Si la respuesta no está en el CONTEXTO, respondé exactamente: "No tengo esa
   información en los documentos disponibles."
3. Cada fragmento indica de qué modelo es: no mezcles datos de un modelo con otro.
4. Respondé en español, en forma breve y directa."""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}")]
)


def _leer_temperature() -> float:
    """
    Se lee del .env en el momento de crear el modelo (no al importar el
    archivo) y se valida: un valor inválido falla con un mensaje claro en
    vez de romper más adelante o ignorarse en silencio.
    """
    valor = os.getenv("TEMPERATURE", "0.2")
    try:
        temperature = float(valor)
    except ValueError:
        raise ValueError(f"TEMPERATURE='{valor}' en el .env no es un número válido")
    if not 0 <= temperature <= 2:
        raise ValueError(f"TEMPERATURE={temperature} fuera de rango (tiene que estar entre 0 y 2)")
    return temperature


def elegir_proveedor() -> str:
    elegido = os.getenv("LLM_PROVIDER", "").strip().lower()
    if elegido:
        if elegido not in PROVEEDORES:
            raise ValueError(f"LLM_PROVIDER='{elegido}' no soportado. Opciones: {list(PROVEEDORES)}")
        if not os.getenv(PROVEEDORES[elegido][0]):
            raise RuntimeError(f"LLM_PROVIDER={elegido}, pero falta {PROVEEDORES[elegido][0]} en el .env")
        return elegido
    for nombre, (variable_key, _, _) in PROVEEDORES.items():
        if os.getenv(variable_key):
            return nombre
    raise RuntimeError(
        "Para usar --responder hace falta la API key de un LLM en el .env: "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY o GEMINI_API_KEY"
    )


def get_llm(proveedor: str):
    variable_key, variable_modelo, modelo_default = PROVEEDORES[proveedor]
    api_key = os.getenv(variable_key)
    modelo = os.getenv(variable_modelo, modelo_default)
    temperature = _leer_temperature()
    logger.info(f"LLM: {proveedor} / {modelo} (temperature={temperature})")

    if proveedor == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=modelo, api_key=api_key, temperature=temperature, timeout=30)
    if proveedor == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=modelo, api_key=api_key, temperature=temperature, timeout=30)
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=modelo, google_api_key=api_key, temperature=temperature, timeout=30)


def _formatear_contexto(docs: list[Document]) -> str:
    bloques = []
    for d in docs:
        pagina = f", página {d.metadata['pagina']}" if d.metadata.get("pagina") else ""
        bloques.append(f"[Fuente: {d.metadata.get('source')}{pagina}]\n{d.page_content}")
    return "\n\n---\n\n".join(bloques)


class Generador:
    """Arma la cadena LCEL una sola vez y la reutiliza en cada pregunta."""

    def __init__(self):
        self.proveedor = elegir_proveedor()
        cadena = prompt | get_llm(self.proveedor) | StrOutputParser()
        self.cadena = cadena.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

    async def responder(self, pregunta: str, docs: list[Document]) -> RespuestaRAG:
        texto = await self.cadena.ainvoke({"contexto": _formatear_contexto(docs), "pregunta": pregunta})
        fuentes, vistas = [], set()
        for d in docs:
            fuente = FuenteCitada(
                source=d.metadata.get("source", "desconocida"),
                pagina=d.metadata.get("pagina"),
                secciones=d.metadata.get("secciones", []),
            )
            if str(fuente) not in vistas:
                vistas.add(str(fuente))
                fuentes.append(fuente)
        return RespuestaRAG(respuesta=texto.strip(), proveedor=self.proveedor, fuentes=fuentes)

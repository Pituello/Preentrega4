"""
Contratos de datos.

ChunkMetadata es la "interfaz de metadatos estricta" que pide la clase
para evitar el Schema Drift: TODO vector que se sube a Pinecone pasa por
este modelo antes del upsert. Si mañana otro script quisiera guardar la
fuente como "file" en vez de "source", Pydantic lo rechaza en el momento
de la ingesta, en vez de que los filtros fallen en silencio meses después.
"""
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 2  # v2: se suman "formato" y "pagina" (soporte PDF)

Categoria = Literal["suv", "crossover", "hatchback", "sedan", "pickup", "sin_categoria"]
Formato = Literal["md", "txt", "pdf"]


class ChunkMetadata(BaseModel):
    """Metadata que viaja junto a cada vector en Pinecone."""

    model_config = {"extra": "forbid"}  # campos no declarados -> error, no se cuelan

    source: str = Field(..., description="Archivo de origen, ej: 'tera.md' (el documento_id)")
    modelo: str = Field(..., description="Modelo de auto en minúsculas, ej: 'tera'")
    categoria: Categoria
    formato: Formato
    pagina: int | None = Field(
        default=None,
        ge=1,
        description="Número de página real (solo PDF). En Markdown/TXT no aplica y se omite",
    )
    secciones: list[str] = Field(
        default_factory=list,
        description="Títulos de sección que abarca el chunk (solo Markdown: en PDF/TXT queda vacía)",
    )
    chunk_index: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=1)
    tokens: int = Field(..., ge=1)
    ingested_at: str = Field(..., description="Fecha de ingesta en formato ISO 8601")
    schema_version: int = SCHEMA_VERSION
    # El texto original va DENTRO de la metadata (paso 2 de la consigna):
    # una sola consulta a Pinecone trae vector + texto + fuente, sin
    # depender de otra base de datos para recuperar el contenido.
    text: str = Field(..., min_length=1)

    def para_pinecone(self) -> dict:
        """
        Metadata lista para el upsert. Pinecone NO acepta valores null en la
        metadata, así que los campos opcionales vacíos (pagina en un .md) se
        omiten en vez de mandarse como None.
        """
        return self.model_dump(exclude_none=True)


class CasoGolden(BaseModel):
    """Una pregunta del benchmark, con el documento que debería recuperarse."""

    pregunta: str = Field(..., min_length=1)
    documento_id_esperado: str = Field(..., min_length=1)


class FuenteCitada(BaseModel):
    """De dónde salió un fragmento usado como contexto (armado por el código, nunca por el LLM)."""

    source: str
    pagina: int | None = None
    secciones: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        if self.pagina:
            return f"{self.source} (pág. {self.pagina})"
        if self.secciones:
            return f"{self.source} ({', '.join(self.secciones[:2])})"
        return self.source


class RespuestaRAG(BaseModel):
    """
    Resultado de la generación opcional (main.py --responder).

    Igual que en la Pre-Entrega 3: el LLM solo redacta `respuesta`. Las
    `fuentes` las arma el código a partir de la metadata REAL de los
    fragmentos recuperados, así el modelo no puede inventar citas.
    """

    respuesta: str = Field(..., min_length=1)
    proveedor: str
    fuentes: list[FuenteCitada]

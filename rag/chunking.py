"""
DocumentProcessor: carga los documentos de /data (.md, .txt y .pdf), los
limpia, los fragmenta por TOKENS (tiktoken) y arma, para cada chunk, su
metadata validada (ChunkMetadata) y un ID determinístico.

Cómo se trata cada formato:
- Markdown: se fragmenta el documento entero; cada chunk guarda las
  secciones (## / ###) que abarca.
- TXT: igual que Markdown, pero sin secciones (no tiene títulos marcados).
- PDF: se fragmenta PÁGINA POR PÁGINA, así cada chunk tiene un número de
  página real en la metadata (el campo "página" que pide la consigna).

"""
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .schemas import SCHEMA_VERSION, ChunkMetadata

# Categoría de cada modelo (etiqueta para filtrar por metadata). Se define
# en un solo lugar a propósito: es parte del esquema, no algo que cada
# script decida por su cuenta.
CATEGORIA_POR_MODELO = {
    "tera": "suv",
    "taos": "suv",
    "nivus": "crossover",
    "polo": "hatchback",
    "virtus": "sedan",
    "vento": "sedan",
    "saveiro": "pickup",
}

FORMATOS = ("md", "txt", "pdf")

_ENCODER = tiktoken.get_encoding("cl100k_base")
_HEADER = re.compile(r"^(#{1,6})\s+(.*)$")


def id_de_chunk(source: str, chunk_index: int, texto: str) -> str:
    """
    ID estable y con PREFIJO por archivo ('tera.md#003-a1b2c3d4'): el
    prefijo permite listar en Pinecone todos los vectores de un archivo.

    El hash incluye la versión del esquema de metadatos: si el esquema
    cambia (ej: se suma un campo), cambian todos los IDs y la sincronización
    reemplaza todos los vectores solos. Si no, los chunks cuyo texto no
    cambió quedarían en Pinecone con la metadata vieja.
    """
    huella = f"v{SCHEMA_VERSION}|{texto}"
    hash_contenido = hashlib.sha256(huella.encode("utf-8")).hexdigest()[:8]
    return f"{source}#{chunk_index:03d}-{hash_contenido}"


def modelo_de(nombre_archivo: str) -> str:
    """
    Modelo de auto a partir del nombre del archivo: 'tera.md' -> 'tera', y
    también 'ficha-tecnica-tera.pdf' -> 'tera'. Si no reconoce ningún
    modelo, usa el nombre del archivo tal cual.
    """
    stem = Path(nombre_archivo).stem.lower()
    for modelo in CATEGORIA_POR_MODELO:
        if modelo in stem:
            return modelo
    return stem


class DocumentProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def clean_text(texto: str) -> str:
        # Los PDF suelen extraer los bullets como caracteres raros (con la
        # fuente de reportlab salen como '\x7f'): se normalizan a "- " al
        # inicio de línea, y se borra cualquier otro carácter de control.
        texto = re.sub(r"^[ \t]*[\x7f•●▪◦][ \t]*", "- ", texto, flags=re.MULTILINE)
        texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", texto)
        texto = re.sub(r"[ \t]+", " ", texto)
        texto = re.sub(r" *\n *", "\n", texto)
        texto = re.sub(r"\n{3,}", "\n\n", texto)
        return texto.strip()

    @staticmethod
    def _titulo(texto: str) -> str:
        for linea in texto.splitlines():
            if linea.strip():
                return linea.strip().lstrip("#").strip()
        return ""

    @staticmethod
    def _mapa_de_secciones(texto: str) -> list[tuple[int, str]]:
        """Lista de (posición en el texto, título de sección) para cada header ## o ###."""
        mapa, posicion = [], 0
        for linea in texto.splitlines(keepends=True):
            m = _HEADER.match(linea.strip())
            if m and len(m.group(1)) >= 2:  # el '#' de nivel 1 es el título del documento
                mapa.append((posicion, m.group(2).strip()))
            posicion += len(linea)
        return mapa

    @staticmethod
    def _secciones_del_chunk(mapa, inicio: int, fin: int) -> list[str]:
        """
        Secciones que abarca el chunk: la que estaba "abierta" cuando
        empieza (aunque el chunk arranque a mitad de sección) + las que
        empiezan dentro de él.
        """
        abierta = None
        dentro = []
        for pos, titulo in mapa:
            if pos <= inicio:
                abierta = titulo
            elif pos < fin:
                dentro.append(titulo)
        secciones = ([abierta] if abierta else []) + dentro
        return list(dict.fromkeys(secciones))  # sin duplicados, respetando el orden

    def _fragmentar(self, texto: str, mapa) -> list[tuple[str, list[str]]]:
        """Divide un texto en chunks y devuelve [(chunk, secciones que abarca)]."""
        resultado, cursor = [], 0
        for pieza in self.splitter.split_text(texto):
            # Posición del chunk dentro del texto, para saber en qué sección
            # arranca. Se calcula a mano: el add_start_index de LangChain a
            # veces devuelve -1 (bug conocido con overlap).
            inicio = texto.find(pieza, cursor)
            if inicio == -1:
                inicio = max(texto.find(pieza), 0)
            cursor = inicio + 1
            resultado.append((pieza, self._secciones_del_chunk(mapa, inicio, inicio + len(pieza))))
        return resultado

    def _piezas_de(self, ruta: Path) -> list[tuple[str, list[str], int | None]]:
        """
        Lee un archivo según su formato y devuelve [(chunk, secciones, página)].
        La página solo existe en los PDF.
        """
        formato = ruta.suffix.lower().lstrip(".")
        if formato == "pdf":
            piezas = []
            for numero, pagina in enumerate(PdfReader(ruta).pages, start=1):
                texto = self.clean_text(pagina.extract_text() or "")
                if not texto:
                    continue  # página vacía o escaneada (sin capa de texto)
                piezas += [(chunk, [], numero) for chunk, _ in self._fragmentar(texto, [])]
            return piezas

        texto = self.clean_text(ruta.read_text(encoding="utf-8"))
        mapa = self._mapa_de_secciones(texto) if formato == "md" else []
        return [(chunk, secciones, None) for chunk, secciones in self._fragmentar(texto, mapa)]

    def _titulo_de_archivo(self, ruta: Path, piezas) -> str:
        """Primera línea con texto del documento (en un PDF, la de la página 1)."""
        return self._titulo(piezas[0][0]) if piezas else ""

    def load_and_process(self, data_dir: str) -> list[tuple[str, Document]]:
        """Devuelve [(id, Document)] con la metadata ya validada por ChunkMetadata."""
        ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rutas = sorted(r for r in Path(data_dir).iterdir() if r.suffix.lower().lstrip(".") in FORMATOS)
        resultado = []

        for ruta in rutas:
            piezas = self._piezas_de(ruta)
            titulo = self._titulo_de_archivo(ruta, piezas)
            modelo = modelo_de(ruta.name)

            for i, (contenido, secciones, pagina) in enumerate(piezas):
                if titulo and titulo not in contenido:
                    contenido = f"{titulo}\n\n{contenido}"

                metadata = ChunkMetadata(
                    source=ruta.name,
                    modelo=modelo,
                    categoria=CATEGORIA_POR_MODELO.get(modelo, "sin_categoria"),
                    formato=ruta.suffix.lower().lstrip("."),
                    pagina=pagina,
                    secciones=secciones,
                    chunk_index=i,
                    total_chunks=len(piezas),
                    tokens=len(_ENCODER.encode(contenido)),
                    ingested_at=ahora,
                    text=contenido,
                )
                doc = Document(page_content=contenido, metadata=metadata.para_pinecone())
                resultado.append((id_de_chunk(ruta.name, i, contenido), doc))

        return resultado

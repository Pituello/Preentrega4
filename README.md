# Pre-Entrega 4 — Sistema RAG Escalable en la Nube con Pinecone

**AI Engineering — Coderhouse**

Módulo de recuperación escalable: indexa documentación técnica en un índice
**Pinecone Serverless** con metadatos estructurados, la recupera con un
**recuperador híbrido** (BM25 + búsqueda vectorial, fusionados con RRF) y
mide su calidad con **Precision@5 y Recall@5** sobre un Golden Set.

Es la evolución de la Pre-Entrega 3: el mismo dataset (siete modelos de
Volkswagen) pasa de ChromaDB local a una base vectorial en la nube. Seis
modelos están en Markdown y uno (la Saveiro) en PDF, para probar la
ingesta con número de página real.

## 1. Resumen del proyecto

| Aspecto | Detalle |
|---|---|
| Vector DB | Pinecone Serverless (AWS `us-east-1`, métrica coseno) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, gratuito, 384 dimensiones) |
| Formatos | Markdown, PDF (página por página) y TXT |
| Chunking | 600 tokens, 100 de overlap (`tiktoken`), dentro del rango 500-800 sugerido |
| Ingesta | Batch upsert asíncrono (lotes de 100) con sincronización del namespace |
| Recuperación | `EnsembleRetriever` = `BM25Retriever` + `PineconeVectorStore` (RRF) |
| Python | 3.12 (`langchain-pinecone` todavía no soporta 3.14) |
| Evaluación | Precision@5, Recall@5 y MRR, comparando 4 configuraciones |
| Clase principal | `RAGSystem` (`rag/retriever.py`) |
| Generación (opcional) | OpenAI, Anthropic o Gemini con `python main.py --responder` |

## 2. Arquitectura

```
/data (*.md, *.pdf, *.txt)
   │
   ▼
DocumentProcessor ─ lectura por formato (PDF: página por página) → limpieza
   │                → chunks de 600 tokens → título antepuesto
   │                → metadata validada con Pydantic (ChunkMetadata)
   ▼
sincronizar_namespace ─ embeddings locales → batch upsert (lotes de 100)
   │                     → borra vectores obsoletos
   ▼
Pinecone Serverless  [índice: volkswagen-rag-hibrido / namespace: vw-modelos-dev]
   │
   ▼
RAGSystem.aretrieve(consulta)
   ├── BM25Retriever (léxico, en memoria)  ─┐
   └── PineconeVectorStore (semántico)     ─┴─► EnsembleRetriever (RRF) → top-5
                  ▲
       filtro opcional por metadata {"modelo": {"$eq": "tera"}}
   │
   ▼ (opcional, --responder)
Generador ─ LLM responde solo con esos 5 fragmentos → RespuestaRAG (texto + fuentes)
```

### Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `rag/config.py` | Configuración central leída del `.env` |
| `rag/embeddings.py` | Modelo de embeddings único; la dimensión se mide, no se hardcodea |
| `rag/schemas.py` | `ChunkMetadata` (esquema estricto de metadatos), `CasoGolden` y `RespuestaRAG` |
| `rag/chunking.py` | `DocumentProcessor`: lectura de MD/TXT/PDF, limpieza, chunking, metadata e IDs determinísticos |
| `rag/pinecone_store.py` | Creación/validación del índice, batch upsert, sincronización y `PineconeVectorStore` |
| `rag/retriever.py` | `RAGSystem`: recuperador híbrido con `EnsembleRetriever` |
| `rag/setup.py` | Fábrica que conecta Pinecone + corpus BM25 en un `RAGSystem` |
| `rag/generation.py` | Generación opcional con LLM (`Generador`) |
| `setup_index.py` | Script de inicialización del índice Serverless |
| `ingest.py` | Pipeline de ingesta |
| `evaluate.py` | Evaluación con Precision@5, Recall@5 y MRR |
| `main.py` | Modo interactivo: fragmentos recuperados y, con `--responder`, respuesta del LLM |
| `golden_set.json` | Benchmark de 8 preguntas con su documento esperado |
| `data/` | Dataset: un documento por modelo (seis `.md` y `saveiro.pdf`) |

## 3. Cómo replicar el índice

1. **Cuenta y API key de Pinecone.** Crear una cuenta gratuita en
   [app.pinecone.io](https://app.pinecone.io) y copiar la API key.

2. **Entorno e instalación.** Requiere **Python 3.12**: ninguna versión de
   `langchain-pinecone` soporta todavía Python 3.14. Se recomienda instalar
   primero PyTorch en versión CPU (mucho más liviana que la versión con
   CUDA, que no se usa):

   ```bash
   py -3.12 -m venv venv             # Windows; en Linux/Mac: python3.12 -m venv venv
   source venv/Scripts/activate      # Windows (Git Bash); en Linux/Mac: source venv/bin/activate
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt
   ```

3. **Variables de entorno.** `cp .env.example .env` y completar
   `PINECONE_API_KEY`. `INDEX_NAME` y `NAMESPACE` tienen valores por defecto.

4. **Crear el índice.**

   ```bash
   python setup_index.py
   ```

   Crea el índice Serverless (`cloud=aws`, `region=us-east-1`, métrica
   coseno, dimensión 384) si no existe. Si ya existe, verifica que su
   dimensión y métrica coincidan con el modelo de embeddings.

5. **Ingesta.**

   ```bash
   python ingest.py            # sincroniza: sube solo lo nuevo o modificado
   python ingest.py --forzar   # vuelve a vectorizar y subir todo
   ```

6. **Evaluación y consulta.**

   ```bash
   python evaluate.py          # reporte de métricas en consola
   python main.py              # consultas interactivas
   python main.py --filtro     # consultas con filtro por modelo
   python main.py --responder  # + respuesta redactada por un LLM (opcional)
   ```

La primera ejecución descarga el modelo de embeddings (~90 MB). La
ingesta, la recuperación y la evaluación no usan ningún LLM: la API key de
OpenAI, Anthropic o Gemini solo hace falta para `--responder`.

## 4. Esquema de metadatos

Cada vector viaja con esta metadata, validada por `ChunkMetadata`
(`extra="forbid"`: un campo no declarado hace fallar la ingesta en vez de
colarse en el índice). Es la "interfaz de metadatos estricta" que la clase
propone contra el *Schema Drift*.

| Campo | Tipo | Ejemplo | Uso |
|---|---|---|---|
| `source` | str | `tera.md` | Documento de origen (el `documento_id` del Golden Set) |
| `modelo` | str | `tera` | Filtro por modelo |
| `categoria` | str | `suv` | Etiqueta de segmento: `suv`, `crossover`, `hatchback`, `sedan`, `pickup` |
| `formato` | str | `pdf` | Formato del documento: `md`, `pdf` o `txt` |
| `pagina` | int | `2` | Número de página real. **Solo en PDF**: en los demás formatos el campo se omite |
| `secciones` | list[str] | `["Colores disponibles", "Garantía"]` | Secciones que abarca el chunk (solo Markdown) |
| `chunk_index` / `total_chunks` | int | `5` / `6` | Posición del chunk dentro del documento |
| `tokens` | int | `317` | Tamaño real del chunk |
| `ingested_at` | str | `2026-09-26T14:07:55+00:00` | Fecha de ingesta (ISO 8601) |
| `schema_version` | int | `2` | Versión del esquema |
| `text` | str | *(contenido)* | Texto original del chunk |

El texto original se guarda **dentro de la metadata** (paso 2 de la
consigna): una sola consulta a Pinecone devuelve vector, texto y fuente,
sin depender de otra base de datos.

## 5. Decisiones de diseño

### 5.1 Dimensión derivada del modelo, no hardcodeada

La consigna menciona 1536 dimensiones, pero ese valor es específico de
`text-embedding-3-small` de OpenAI. Este sistema usa un modelo local de 384
dimensiones, y `get_dimension()` la **mide** generando un embedding de
prueba. `setup_index()` compara ese valor contra el índice existente y
falla con un mensaje explícito si no coinciden — el *mismatch de
dimensiones* se detecta antes de subir un solo vector.

### 5.2 Namespace para aislar datos

Todos los vectores van al namespace `vw-modelos-dev`. Permite convivir en
el mismo índice con otros datasets o entornos (por ejemplo, un
`vw-modelos-prod`) sin mezclar resultados.

### 5.3 Batch upsert asíncrono

Los chunks se vectorizan y suben en lotes de `BATCH_SIZE` (100, el "punto
dulce" que marca la clase), hasta 4 lotes en paralelo con
`asyncio.gather` y un semáforo. El embedding local (CPU) y el upsert (red)
son síncronos, así que se ejecutan con `asyncio.to_thread` para no
bloquear el event loop.

### 5.4 Ingesta idempotente y sincronizada

Cada chunk tiene un ID determinístico con prefijo por archivo
(`tera.md#005-c5245a9d`: archivo, posición y hash del contenido).
`sincronizar_namespace()` compara los IDs de `/data` con los que ya están
en Pinecone y:

- sube solo los chunks nuevos o modificados (un hash distinto es un ID distinto);
- borra los que ya no existen (archivos eliminados o versiones viejas de chunks modificados);
- no vuelve a vectorizar lo que no cambió.

Esto resuelve una limitación que quedó pendiente en la Pre-Entrega 3,
donde un cambio de contenido dentro de un archivo exigía borrar el índice
y reindexar todo. Verificado con un índice simulado: al agregar una
sección a `tera.md` y borrar `vento.md`, la ingesta subió 1 chunk, borró 6
y dejó el namespace idéntico a `/data`.

### 5.5 Título del documento en cada chunk

Cada chunk se vectoriza de forma aislada. Si no repite el nombre del
modelo, la búsqueda no lo asocia con ese modelo: en la Pre-Entrega 3, la
pregunta *"¿en qué colores viene el Tera?"* devolvía los colores de la
Saveiro por esa razón. Por eso el título del documento se antepone a cada
chunk antes de vectorizarlo.

### 5.6 BM25 con preprocesamiento para español

El `BM25Retriever` de LangChain tokeniza por defecto con `texto.split()`:
"¿Tera?" no coincide con "Tera", ni "garantía" con "Garantía".
`tokenizar()` pasa a minúsculas, quita tildes, extrae solo palabras y
números, y descarta *stopwords* frecuentes.

### 5.7 Fusión con RRF, no suma de scores

BM25 produce puntajes sin techo fijo; la similitud coseno va de -1 a 1.
Sumarlos directamente es uno de los errores que marca la consigna.
`EnsembleRetriever` fusiona con *Reciprocal Rank Fusion*, que combina por
**posición** en cada ranking. Pesos por defecto: 0,5 / 0,5, configurables
con `BM25_WEIGHT` y `VECTOR_WEIGHT`.

### 5.8 Corpus de BM25 en memoria

BM25 necesita todo el corpus en memoria. Se reconstruye desde `/data` con
el mismo `DocumentProcessor` de la ingesta, por lo que los textos son
idénticos a los de Pinecone. Como eso solo es cierto si la última ingesta
fue posterior al último cambio en `/data`, `crear_rag_system()` compara la
cantidad de vectores del namespace con la de chunks locales y avisa si no
coinciden.

### 5.9 Filtro opcional por metadata

Con `filtrar_por_modelo=True`, si la consulta nombra un único modelo,
`RAGSystem` acota la búsqueda vectorial con `{"modelo": {"$eq": "tera"}}` y
usa un BM25 construido solo con ese modelo. Es la solución que la
Pre-Entrega 3 dejó documentada como pendiente. Tiene un costo, analizado
en la sección 6.

### 5.10 Búsquedas asíncronas concurrentes

Al correr `evaluate.py` contra Pinecone real apareció un bug de
`langchain-pinecone` 0.2.x: todas las búsquedas asíncronas de
`PineconeVectorStore` comparten **una sola sesión HTTP**, y cada búsqueda
la cierra al terminar. Con varias consultas en paralelo (`asyncio.gather`),
la primera en terminar cierra la sesión mientras las demás la siguen
usando: `RuntimeError: Session is closed`. Con una consulta por vez
(`main.py`) no se manifiesta, por eso tarda en aparecer.

`PineconeVectorStoreConcurrente` (en `rag/pinecone_store.py`) resuelve las
búsquedas asíncronas ejecutando el camino síncrono en un thread
(`asyncio.to_thread`). El cliente síncrono de Pinecone usa un pool de
conexiones que sí soporta consultas concurrentes, así que la evaluación
sigue lanzando las preguntas en paralelo. El bug se reprodujo con un
índice simulado (4 consultas en paralelo: falla) y el arreglo se verificó
con 20 consultas en paralelo.

### 5.11 Soporte para PDF con número de página

`DocumentProcessor` lee Markdown, TXT y PDF. Los PDF se leen con `pypdf` y
se fragmentan **página por página**, así cada chunk guarda su número de
página real en `pagina` (el metadato de "página" que pide la consigna). Tres
detalles que aparecieron al implementarlo:

- **Pinecone no acepta `null` en la metadata.** `pagina` no aplica a los
  Markdown, así que se omite en vez de mandarse vacía
  (`ChunkMetadata.para_pinecone()` usa `exclude_none=True`).
- **Los bullets de un PDF se extraen como caracteres de control**
  (`\x7f` con la fuente del PDF de este dataset). `clean_text()` los
  normaliza a `- `, igual que en Markdown.
- **Cambiar el esquema obliga a reemplazar todos los vectores.** Los chunks
  cuyo texto no cambió conservan su ID, así que la sincronización no los
  volvería a subir y quedarían en Pinecone con la metadata vieja. Por eso
  el hash del ID incluye `SCHEMA_VERSION`: al pasar a la versión 2, todos
  los IDs cambian y la ingesta reemplaza todo sola.

`saveiro.pdf` es una ficha de 3 páginas generada a partir del contenido
verificado de la Saveiro. Reemplazó a `saveiro.md` para no tener el mismo
texto dos veces en el índice, lo que ensuciaría la precisión.

### 5.12 Generación opcional

La consigna evalúa la recuperación, no la generación. Aun así,
`python main.py --responder` le pasa los top-5 fragmentos a un LLM
(OpenAI, Anthropic o Gemini, según las keys del `.env`) con un prompt que
lo obliga a responder solo con ese contexto. Igual que en la Pre-Entrega 3,
el LLM solo redacta el texto: las fuentes citadas (con página, si vienen de
un PDF) las arma el código a partir de la metadata real.

`TEMPERATURE` se lee del `.env` **al crear el modelo**, no al importar el
módulo, y se valida (número entre 0 y 2). Así un valor inválido falla con
un mensaje claro, y el valor configurado se respeta aunque el módulo se
importe antes de cargar el `.env`.

## 6. Resultados de la evaluación

Resultados obtenidos con `python evaluate.py` contra el índice real de
Pinecone Serverless (8 preguntas, k=5, namespace `vw-modelos-dev`):

| Configuración | Recall@5 | Precision@5 | MRR |
|---|---|---|---|
| Vectorial (Pinecone) | 100,0% | 40,0% | 0,71 |
| Léxica (BM25) | 100,0% | 35,0% | 0,91 |
| **Híbrida (BM25 + Pinecone)** | **100,0%** | **37,5%** | **0,92** |
| Híbrida + filtro por modelo | 87,5% | 75,0% | 0,88 |
| *Techo teórico de Precision@5* | | *95,0%* | |

### Análisis

- **La nube dio exactamente lo mismo que la validación local.** Antes de
  conectar Pinecone, el sistema se validó con una base vectorial en memoria
  (mismo modelo de embeddings, mismos chunks). Los resultados contra
  Pinecone real fueron idénticos en las cuatro configuraciones y pregunta
  por pregunta: con 33 vectores, la búsqueda aproximada (ANN) de Pinecone
  devuelve lo mismo que una búsqueda exacta.
- **Recall@5 no discrimina en este benchmark.** Las tres configuraciones
  sin filtro encuentran el documento esperado en las 8 preguntas: con solo
  7 documentos, es relativamente fácil que el correcto entre en un top-5.
  Por eso se agregó **MRR**, que mide *en qué posición* aparece.
- **El híbrido gana en MRR (0,92).** La búsqueda vectorial sola encuentra
  todo, pero a menudo no en el primer lugar (0,71). Combinarla con BM25
  sube el documento correcto al primer puesto en 7 de 8 preguntas.
- **El caso del Tera quedó resuelto.** *"¿En qué colores viene el
  Tera?"* —la pregunta que fallaba en la Pre-Entrega 3— ahora devuelve
  `tera.md` en el primer puesto.
- **El filtro por modelo es un trade-off, no una mejora gratis.** Duplica
  la precisión (37,5% → 75%) porque elimina el ruido de otros
  modelos, pero falla en *"¿qué capacidad de baúl tiene la versión sedán
  del Polo?"*: detecta "Polo", filtra por `polo.md` y excluye a `virtus.md`,
  que tiene la respuesta. Es el riesgo de los *hard filters* que menciona
  la clase: si el filtro se infiere mal, el documento correcto queda fuera
  de la búsqueda. Por eso está desactivado por defecto.
- **La precisión está limitada por diseño.** Con un solo documento
  relevante por pregunta y documentos de 4 a 6 chunks, el máximo teórico
  es 95%, no 100%.

## 7. Cumplimiento de la consigna

| Requisito | Dónde |
|---|---|
| `.env` con `PINECONE_API_KEY`, `OPENAI_API_KEY` (o Anthropic) e `INDEX_NAME` | `.env.example`, `rag/config.py`, `rag/generation.py` |
| Script que verifica si el índice existe y lo crea (Serverless) | `setup_index.py` |
| Carga de documentos (PDF, Markdown) | `data/`, `rag/chunking.py` |
| Chunking con `RecursiveCharacterTextSplitter` | `rag/chunking.py` |
| Embeddings en Pinecone con contenido y fuente en la metadata | `rag/pinecone_store.py` |
| Metadatos avanzados (fuente, página, categoría) | `rag/schemas.py` |
| `BM25Retriever` combinado con Pinecone vía `EnsembleRetriever` | `rag/retriever.py` |
| Clase `RAGSystem` que devuelve el top-5 | `rag/retriever.py` |
| Benchmark de al menos 5 preguntas | `golden_set.json` (8 preguntas) |
| `evaluate.py` con Recall@5 y Precision@5 | `evaluate.py` |
| Reporte en consola | `evaluate.py` |
| Instrucciones para replicar el índice | Sección 3 de este README |

### Extensiones no requeridas por la consigna

- Esquema de metadatos validado con Pydantic (anti *Schema Drift*)
- Validación de dimensión y métrica de un índice existente
- Ingesta sincronizada: sube solo lo modificado y borra lo obsoleto
- Batch upsert asíncrono con concurrencia limitada
- Tokenización de BM25 adaptada al español
- Filtro opcional por modelo con metadatos de Pinecone
- MRR y comparación de cuatro configuraciones de recuperación
- Chequeo de consistencia entre el corpus de BM25 y el namespace de Pinecone
- Soporte para PDF con número de página real, además de Markdown y TXT
- Generación opcional con LLM (`--responder`) y fuentes citadas con página
- Arreglo del bug de sesión compartida de `langchain-pinecone` en consultas concurrentes

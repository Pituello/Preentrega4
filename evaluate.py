"""
Evaluación del recuperador sobre el Golden Set (golden_set.json).

Métricas por pregunta (con k = TOP_K = 5):
- Recall@5: ¿el documento esperado aparece entre los 5 chunks recuperados?
  Como cada pregunta tiene UN solo documento relevante, da 0 o 1.
- Precision@5: de los 5 chunks recuperados, qué fracción viene del
  documento esperado. Ojo: si ese documento tiene menos de 5 chunks, la
  precisión máxima posible es menor a 100% (se reporta el techo).
- MRR (extra, no lo pide la consigna): 1 / posición del primer chunk
  relevante. Premia que el correcto aparezca arriba, no solo "en algún lado".

Se corren 4 configuraciones para poder COMPARAR, no solo medir:
vectorial sola, BM25 sola, híbrido, e híbrido + filtro por modelo.

Uso:
    python evaluate.py
"""
import asyncio
import json
from collections import Counter

from rag import config
from rag.schemas import CasoGolden
from rag.setup import crear_rag_system

CONFIGURACIONES = [
    ("Vectorial (Pinecone)", {"modo": "vectorial"}),
    ("Léxica (BM25)", {"modo": "bm25"}),
    ("Híbrida (BM25 + Pinecone)", {"modo": "hibrido"}),
    ("Híbrida + filtro por modelo", {"modo": "hibrido", "filtrar_por_modelo": True}),
]


def cargar_golden_set(ruta: str = "golden_set.json") -> list[CasoGolden]:
    with open(ruta, encoding="utf-8") as f:
        return [CasoGolden(**caso) for caso in json.load(f)]


def metricas_de_un_caso(fuentes: list[str], esperado: str, k: int) -> dict:
    relevantes = sum(1 for f in fuentes if f == esperado)
    posicion = next((i for i, f in enumerate(fuentes, 1) if f == esperado), None)
    return {
        "recall": 1.0 if posicion else 0.0,
        "precision": relevantes / k,
        "mrr": 1.0 / posicion if posicion else 0.0,
    }


async def evaluar(rag, golden: list[CasoGolden], k: int) -> dict:
    resultados = await asyncio.gather(*(rag.aretrieve(c.pregunta) for c in golden))
    detalle = []
    for caso, docs in zip(golden, resultados):
        fuentes = [d.metadata.get("source") for d in docs]
        detalle.append({"caso": caso, "fuentes": fuentes, **metricas_de_un_caso(fuentes, caso.documento_id_esperado, k)})

    n = len(detalle)
    return {
        "detalle": detalle,
        "recall": sum(d["recall"] for d in detalle) / n,
        "precision": sum(d["precision"] for d in detalle) / n,
        "mrr": sum(d["mrr"] for d in detalle) / n,
    }


def techo_de_precision(golden: list[CasoGolden], chunks_por_doc: Counter, k: int) -> float:
    """Precision@k máxima alcanzable dado cuántos chunks tiene cada documento esperado."""
    return sum(min(chunks_por_doc[c.documento_id_esperado], k) / k for c in golden) / len(golden)


async def main():
    golden = cargar_golden_set()
    k = config.TOP_K
    reportes = []

    for nombre, opciones in CONFIGURACIONES:
        rag = crear_rag_system(k=k, **opciones)
        reportes.append((nombre, await evaluar(rag, golden, k)))

    chunks_por_doc = Counter(d.metadata["source"] for d in rag._bm25_global.docs)
    techo = techo_de_precision(golden, chunks_por_doc, k)

    print("\n" + "=" * 86)
    print(f"EVALUACIÓN — {len(golden)} preguntas, k={k}, namespace '{config.NAMESPACE}'")
    print("=" * 86)
    print(f"{'Configuración':<32}{'Recall@' + str(k):>14}{'Precision@' + str(k):>16}{'MRR':>10}")
    print("-" * 86)
    for nombre, r in reportes:
        print(f"{nombre:<32}{r['recall']:>14.1%}{r['precision']:>16.1%}{r['mrr']:>10.2f}")
    print("-" * 86)
    print(f"{'Techo teórico de Precision@' + str(k):<32}{'':>14}{techo:>16.1%}")
    print("  (el máximo posible: algunos documentos tienen menos de 5 chunks)")

    print("\n" + "=" * 86)
    print("DETALLE — Híbrida (BM25 + Pinecone)")
    print("=" * 86)
    hibrido = dict(reportes)["Híbrida (BM25 + Pinecone)"]
    for d in hibrido["detalle"]:
        estado = "✅" if d["recall"] else "❌"
        print(f"{estado} {d['caso'].pregunta}")
        print(f"   Esperado: {d['caso'].documento_id_esperado} | Recuperados: {d['fuentes']}")
        print(f"   Recall@{k}: {d['recall']:.0%} | Precision@{k}: {d['precision']:.0%} | RR: {d['mrr']:.2f}\n")

    # Diferencias entre configuraciones, pregunta por pregunta
    print("=" * 86)
    print("¿DÓNDE DIFIEREN LAS CONFIGURACIONES? (✅ = recuperó el documento esperado)")
    print("=" * 86)
    nombres_cortos = ["Vect", "BM25", "Híbr", "Filt"]
    print(f"{'':<4}" + "".join(f"{n:>6}" for n in nombres_cortos) + "  Pregunta")
    for i, caso in enumerate(golden):
        marcas = "".join(f"{'✅' if r['detalle'][i]['recall'] else '❌':>5} " for _, r in reportes)
        print(f"{i + 1:<4}{marcas}  {caso.pregunta[:60]}")


if __name__ == "__main__":
    asyncio.run(main())

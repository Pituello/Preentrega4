"""
Modo interactivo: consultá el recuperador híbrido y mirá qué fragmentos
trae (fuente, página o sección, extracto).

Uso:
    python main.py                 # híbrido: muestra los top-5 fragmentos
    python main.py --filtro        # + filtro por modelo en la metadata
    python main.py --responder     # + respuesta redactada por un LLM (opcional,
                                   #   necesita una API key de LLM en el .env)

Las opciones se pueden combinar: python main.py --filtro --responder
"""
import asyncio
import sys

from rag.setup import crear_rag_system


def _ubicacion(metadata: dict) -> str:
    if metadata.get("pagina"):
        return f"página {metadata['pagina']}"
    return ", ".join(metadata.get("secciones", [])[:3])


async def main():
    filtro = "--filtro" in sys.argv
    responder = "--responder" in sys.argv

    rag = crear_rag_system(filtrar_por_modelo=filtro)
    generador = None
    if responder:
        from rag.generation import Generador

        generador = Generador()  # falla acá, con mensaje claro, si no hay API key de LLM

    print("💬 Recuperador híbrido (BM25 + Pinecone) — modelos Volkswagen")
    print(f"   Filtro por modelo: {'activado' if filtro else 'desactivado'} | "
          f"Respuesta con LLM: {generador.proveedor if generador else 'desactivada'}")
    print("   Escribí 'salir' para terminar\n")

    while True:
        consulta = input("🧑 Vos: ").strip()
        if consulta.lower() in ("salir", "exit", "quit", ""):
            print("\n👋 Listo, terminamos la sesión.")
            break

        if filtro:
            modelo = rag.detectar_modelo(consulta)
            print(f"   (filtro aplicado: {modelo or 'ninguno, la consulta no nombra un único modelo'})")

        docs = await rag.aretrieve(consulta)
        print(f"\n📎 Top {len(docs)} fragmentos:")
        for i, d in enumerate(docs, 1):
            extracto = d.page_content.replace("\n", " ")[:160]
            print(f"\n  {i}. [{d.metadata.get('source')}] {_ubicacion(d.metadata)}")
            print(f"     {extracto}...")

        if generador:
            resultado = await generador.responder(consulta, docs)
            print(f"\n🤖 Respuesta ({resultado.proveedor}):\n{resultado.respuesta}")
            print(f"\n📚 Fuentes: {', '.join(str(f) for f in resultado.fuentes)}")
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import logging
from norm_rag.normlightrag_openai import NormaRAG

async def main():
    # Configura il logging a livello di modulo
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    try:
        # Inizializza NormaRAG
        norma_rag = NormaRAG(
            base_url="http://127.0.0.1:5000/fetch_all_data",
            working_dir="./output/penale_out_v2(cp+cpp+man)",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            verbose=True,
            max_chunk_tokens=8192,  
            requests_per_minute=450,
            tokens_per_minute=199000
        )

        # Percorsi dei file aggiuntivi
        additional_files = [
            "src/data/penale/1.txt",
            "src/data/penale/2.txt",
            "src/data/penale/3.txt",
            "src/data/penale/4.txt",
            "src/data/penale/5.txt",
        ]

        # Dettagli degli atti da processare
        act_details = [
            {
                "act_type": "codice penale",
                "date": "",
                "act_number": "",
                "version": "vigente",
                "version_date": "2024-10-31",
                "show_brocardi_info": True,
                "articles": (1, 734),
                "batch_size": 50,
                "atomize": True,  # Se True, salva ogni articolo in un file separato
                "save": False,    # Se False, restituisce i contenuti in memoria
            },
            {
                "act_type": "codice di procedura penale",
                "date": "",
                "act_number": "",
                "version": "vigente",
                "version_date": "2024-10-31",
                "show_brocardi_info": True,
                "articles": (1, 746),
                "batch_size": 50,
                "atomize": True,  # Se True, salva ogni articolo in un file separato
                "save": False,    # Se False, restituisce i contenuti in memoria
            }
            # Aggiungi altri atti se necessario
        ]

        # Avvia il processo principale
        await norma_rag.main_process(
            additional_files=additional_files,
            act_details=act_details
        )

    except Exception as e:
        logging.error(f"Si è verificato un errore nel main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

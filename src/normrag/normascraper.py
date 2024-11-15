import aiohttp
from aiocache import cached
import asyncio
import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Union, List, Tuple

class NormaScraper:
    def __init__(self, base_url="http://127.0.0.1:5000/fetch_all_data"):
        self.base_url = base_url
        self.session = None

    @retry(
        retry=retry_if_exception_type(aiohttp.ClientError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60)
    )
    @cached(ttl=3600)
    async def fetch_data(self, url, data):
        logging.debug(f"Inviando richiesta POST per l'articolo/intervallo {data['article']}...")
        async with self.session.post(url, json=data) as response:
            logging.debug(f"Ricevuta risposta per {data['article']} con codice di stato: {response.status}")
            response.raise_for_status()
            return await response.json()

    def extract_relevant_data(self, article_data):
        article_text = article_data.get("article_text", "Testo non disponibile")
        norma_data = article_data.get("norma_data", {})
        brocardi_info = article_data.get("brocardi_info", {})

        norma_details = {
            "Tipo_atto": norma_data.get("tipo_atto"),
            "Data": norma_data.get("data"),
            "Numero_Atto": norma_data.get("numero_atto"),
            "Articolo": norma_data.get("numero_articolo")
        }
        norma_details = {k: v for k, v in norma_details.items() if v}

        brocardi_details = {}
        for key, value in brocardi_info.items():
            if value and key.lower() != "link":
                formatted_key = key.capitalize()
                if key in ["Massime", "Brocardi"] and isinstance(value, list):
                    brocardi_details[formatted_key] = "\n- " + "\n- ".join(value)
                else:
                    brocardi_details[formatted_key] = value

        position = brocardi_info.get("position", "")
        return article_text, norma_details, brocardi_details, position

    def write_article_content(self, article_text, norma_details, brocardi_details, position):
        content_lines = [
            '\n' + '='*50 + '\n',
            f"\n===== Articolo {norma_details.get('Articolo', 'Sconosciuto')} =====\n",
            "[Testo Principale]\n",
            f"{article_text}\n"
        ]

        if position:
            content_lines.append(f"\nPosizione: {position}\n")

        content_lines.append("\n[Dettagli Norma]\n")
        for key, value in norma_details.items():
            content_lines.append(f"{key}: {value}\n")

        if brocardi_details:
            content_lines.append("\n[Informazioni Secondarie - Brocardi]\n")
            for key, value in brocardi_details.items():
                content_lines.append(f"{key}: {value}\n")

        return ''.join(content_lines)

    async def fetch_and_save_articles(
        self,
        act_type: str,
        date: str,
        act_number: str,
        version: str,
        version_date: str,
        show_brocardi_info: bool,
        articles: Union[List[int], Tuple[int, int]],
        atomize: bool = False,
        save: bool = True,
        file_name: str = "output.txt"
    ):
        if save:
            if atomize:
                folder_name = "_".join([word[:3] for word in act_type.split()])
                folder_path = os.path.join("data", folder_name)
                os.makedirs(folder_path, exist_ok=True)
            else:
                folder_path = os.path.join("data", file_name)
                os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        else:
            results = []

        data_template = {
            "act_type": act_type,
            "date": date,
            "act_number": act_number,
            "version": version,
            "version_date": version_date,
            "show_brocardi_info": show_brocardi_info
        }

        article_ranges = self.create_article_ranges(articles, batch_size=50)

        async with aiohttp.ClientSession() as session:
            self.session = session
            for article_range in article_ranges:
                data = data_template.copy()
                data["article"] = article_range

                try:
                    articles_data = await self.fetch_data(self.base_url, data)
                    logging.info(f"Articoli {article_range} recuperati con successo.")

                    for article in articles_data:
                        article_text, norma_details, brocardi_details, position = self.extract_relevant_data(article)
                        content = self.write_article_content(article_text, norma_details, brocardi_details, position)

                        if save:
                            if atomize:
                                article_number = norma_details.get("Articolo", "Sconosciuto")
                                file_path = os.path.join(folder_path, f"{article_number}.txt")
                                with open(file_path, "w", encoding="utf-8") as file:
                                    file.write(content)
                                logging.debug(f"Articolo {article_number} salvato in {file_path}.")
                            else:
                                with open(folder_path, "a", encoding="utf-8") as file:
                                    file.write(content)
                                logging.debug(f"Articoli {article_range} salvati in {folder_path}.")
                        else:
                            results.append(content)

                except aiohttp.ClientError as e:
                    logging.error(f"Errore durante la richiesta dell'intervallo {article_range}: {e}")

                # Opzionale: gestire il rate limit se necessario
                # await asyncio.sleep(2)

        if not save:
            return results

    def create_article_ranges(self, articles: Union[List[int], Tuple[int, int]], batch_size: int) -> List[str]:
        if isinstance(articles, list):
            return [str(article) for article in articles]
        elif isinstance(articles, tuple):
            start, end = articles
            return [f"{i}-{min(i + batch_size - 1, end)}" for i in range(start, end + 1, batch_size)]
        else:
            raise ValueError("Il parametro 'articles' deve essere una lista di numeri di articoli o una tupla di range.")

import asyncio
import os
import logging
import time
from typing import List, Union, Tuple, Optional

import aiohttp
from aiocache import cached, Cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import numpy as np
import tiktoken
from lightrag.lightrag import LightRAG, EmbeddingFunc
from lightrag.llm import openai_complete_if_cache, openai_embedding
from openai import RateLimitError, APIError


class NormaRAG:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5000/fetch_all_data",
        working_dir: str = "./output",
        model_name: str = 'gpt-4o-mini',
        embedding_model: str = 'text-embedding-3-small',
        verbose: bool = True,
        max_chunk_tokens: Optional[int] = None,
        requests_per_minute: int = 450,
        tokens_per_minute: int = 199000
    ):
        """
        Inizializza la classe NormaRAG.

        :param base_url: URL di base per lo scraping degli articoli.
        :param working_dir: Directory di lavoro per salvare output e progressi.
        :param model_name: Nome del modello LLM da utilizzare.
        :param embedding_model: Nome del modello di embedding da utilizzare.
        :param verbose: Abilita o disabilita il logging dettagliato.
        :param max_chunk_tokens: Numero massimo di token per chunk di testo.
        :param requests_per_minute: Numero massimo di richieste per minuto.
        :param tokens_per_minute: Numero massimo di token per minuto.
        """
        self.base_url = base_url
        self.working_dir = working_dir
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.verbose = verbose
        self.encoding = tiktoken.get_encoding("cl100k_base")
        os.makedirs(self.working_dir, exist_ok=True)

        # Configurazione del logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )
        logging.info("Inizializzazione di NormaRAG.")

        # Limiti di rate
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.min_interval = 60.0 / self.requests_per_minute
        self.last_request_time = 0
        self.tokens_used_in_current_minute = 0
        self.minute_start_time = time.time()

        # Limite massimo di token per chunk
        self.max_chunk_tokens = (self.tokens_per_minute // 2) if max_chunk_tokens is None else max_chunk_tokens

        # Inizializzazione di LightRAG
        self.base_rag = self._initialize_rag()

        # Sessione aiohttp
        self.session = None

    def _initialize_rag(self) -> LightRAG:
        """
        Inizializza l'istanza di LightRAG.

        :return: Istanza di LightRAG.
        """
        logging.debug("Inizializzazione di LightRAG.")
        return LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self.llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,
                max_token_size=8192,
                func=self.embedding_func,
            ),
        )

    @retry(
        retry=retry_if_exception_type(aiohttp.ClientError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60)
    )
    @cached(ttl=3600, cache=Cache.MEMORY)  # Utilizzo della cache in memoria
    async def fetch_data(self, data: dict) -> List[dict]:
        """
        Effettua una richiesta POST per recuperare i dati degli articoli.

        :param data: Dati da inviare nella richiesta POST.
        :return: Lista di dizionari con i dati degli articoli.
        """
        logging.debug(f"Inviando richiesta POST per l'articolo/intervallo {data['article']}...")
        async with self.session.post(self.base_url, json=data) as response:
            logging.debug(f"Ricevuta risposta per {data['article']} con codice di stato: {response.status}")
            response.raise_for_status()
            return await response.json()

    def extract_relevant_data(self, article_data: dict) -> tuple:
        """
        Estrae il testo dell'articolo e altre informazioni dai dati ricevuti.

        :param article_data: Dati grezzi dell'articolo.
        :return: Tuple contenente article_text, norma_details, brocardi_details, position.
        """
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

    def write_article_content(self, article_text: str, norma_details: dict, brocardi_details: dict, position: str) -> str:
        """
        Crea una stringa formattata contenente tutte le informazioni rilevanti dell'articolo.

        :param article_text: Testo principale dell'articolo.
        :param norma_details: Dettagli della norma.
        :param brocardi_details: Informazioni Brocardi.
        :param position: Posizione dell'articolo.
        :return: Stringa formattata.
        """
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

    async def llm_model_func(self, prompt: str, **kwargs) -> str:
        """
        Funzione wrapper per interagire con il modello LLM di OpenAI.

        :param prompt: Prompt da inviare al modello.
        :return: Risposta del modello.
        """
        try:
            await self._wait_for_rate_limit()
            input_tokens = len(self.encoding.encode(prompt))
            if input_tokens > self.max_chunk_tokens:
                logging.error(f"I token del prompt ({input_tokens}) superano il limite massimo ({self.max_chunk_tokens}).")
                raise ValueError("I token del prompt superano il massimo consentito per richiesta.")
            logging.debug(f"Invio del prompt di {input_tokens} token al modello LLM.")

            response = await openai_complete_if_cache(
                self.model_name,
                prompt,
                api_key=os.getenv("OPENAI_API_KEY"),
                **kwargs
            )
            output_tokens = len(self.encoding.encode(response))
            total_tokens = input_tokens + output_tokens
            logging.info(f"Richiesta LLM riuscita. Token utilizzati in questa richiesta: {total_tokens}")
            self._update_token_usage(total_tokens)
            return response
        except (RateLimitError, APIError) as e:
            logging.error(f"Errore di rate limit in llm_model_func: {e}")
            await self._handle_rate_limit_error()
            return await self.llm_model_func(prompt, **kwargs)
        except Exception as e:
            logging.error(f"Errore in llm_model_func: {e}")
            raise

    async def embedding_func(self, texts: List[str]) -> np.ndarray:
        """
        Funzione wrapper per generare embeddings utilizzando il modello di embedding di OpenAI.

        :param texts: Lista di testi da trasformare in embeddings.
        :return: Array NumPy di embeddings.
        """
        try:
            await self._wait_for_rate_limit()
            tokens_used = sum(len(self.encoding.encode(text)) for text in texts)
            logging.info(f"Generazione degli embeddings per {len(texts)} testi. Token utilizzati: {tokens_used}")
            embeddings = await openai_embedding(
                texts,
                model=self.embedding_model,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            logging.debug("Embeddings generati con successo.")
            self._update_token_usage(tokens_used)
            return embeddings
        except (RateLimitError, APIError) as e:
            logging.error(f"Errore di rate limit in embedding_func: {e}")
            await self._handle_rate_limit_error()
            return await self.embedding_func(texts)
        except Exception as e:
            logging.error(f"Errore in embedding_func: {e}")
            raise

    async def process_text(self, text: str, norma_details: Optional[dict] = None, brocardi_details: Optional[dict] = None, position: Optional[str] = None):
        """
        Processa un singolo testo (articolo o documento aggiuntivo).

        :param text: Testo da processare.
        :param norma_details: Dettagli della norma.
        :param brocardi_details: Informazioni Brocardi.
        :param position: Posizione dell'articolo.
        """
        if norma_details and brocardi_details and position:
            content = self.write_article_content(text, norma_details, brocardi_details, position)
        else:
            content = text  # Per documenti aggiuntivi senza dettagli specifici

        chunks = self.split_text_into_chunks(content)
        if not chunks:
            logging.warning("Nessun chunk da elaborare per questo testo.")
            return
        for i, chunk in enumerate(chunks):
            logging.debug(f"Elaborazione del chunk {i + 1}/{len(chunks)}.")
            await self.base_rag.ainsert(chunk)
        logging.info("Chunk elaborati con successo.")

    def save_article_content(self, content: str, folder_path: str, article_number: str):
        """
        Salva il contenuto formattato dell'articolo in un file.

        :param content: Contenuto formattato dell'articolo.
        :param folder_path: Percorso della cartella dove salvare il file.
        :param article_number: Numero dell'articolo.
        """
        file_path = os.path.join(folder_path, f"{article_number}.txt")
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
            logging.debug(f"Articolo {article_number} salvato in {file_path}.")
        except Exception as e:
            logging.error(f"Errore durante il salvataggio del file {file_path}: {e}")

    async def process_articles(
        self,
        act_type: str,
        date: str,
        act_number: str,
        version: str,
        version_date: str,
        show_brocardi_info: bool,
        articles: Union[List[int], Tuple[int, int]],
        batch_size: int = 50,
        atomize: bool = False,
        save: bool = True,
        file_name: str = "output.txt"
    ):
        """
        Recupera e processa gli articoli specificati.

        :param act_type: Tipo di atto (es. "codice penale").
        :param date: Data dell'atto.
        :param act_number: Numero dell'atto.
        :param version: Versione dell'atto.
        :param version_date: Data della versione.
        :param show_brocardi_info: Flag per mostrare informazioni Brocardi.
        :param articles: Lista di numeri di articoli o tupla di range.
        :param batch_size: Numero di articoli per batch.
        :param atomize: Se True, salva ogni articolo in un file separato.
        :param save: Se True, salva i file; altrimenti, restituisce i contenuti.
        :param file_name: Nome del file di output se save=True e atomize=False.
        :return: Lista di contenuti se save=False, altrimenti None.
        """
        data_template = {
            "act_type": act_type,
            "date": date,
            "act_number": act_number,
            "version": version,
            "version_date": version_date,
            "show_brocardi_info": show_brocardi_info
        }

        article_ranges = self.create_article_ranges(articles, batch_size)
        logging.info(f"Articoli suddivisi in {len(article_ranges)} intervalli.")

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

        async with aiohttp.ClientSession() as session:
            self.session = session
            for article_range in article_ranges:
                data = data_template.copy()
                data["article"] = article_range
                try:
                    articles_data = await self.fetch_data(data)
                    logging.info(f"Articoli {article_range} recuperati con successo.")

                    for article in articles_data:
                        article_text, norma_details, brocardi_details, position = self.extract_relevant_data(article)
                        await self.process_text(article_text, norma_details, brocardi_details, position)

                        if save:
                            if atomize:
                                article_number = norma_details.get("Articolo", "Sconosciuto")
                                content = self.write_article_content(article_text, norma_details, brocardi_details, position)
                                self.save_article_content(content, folder_path, article_number)
                            else:
                                content = self.write_article_content(article_text, norma_details, brocardi_details, position)
                                try:
                                    with open(folder_path, "a", encoding="utf-8") as file:
                                        file.write(content)
                                    logging.debug(f"Articoli {article_range} salvati in {folder_path}.")
                                except Exception as e:
                                    logging.error(f"Errore durante il salvataggio degli articoli {article_range} in {folder_path}: {e}")
                        else:
                            content = self.write_article_content(article_text, norma_details, brocardi_details, position)
                            results.append(content)

                except aiohttp.ClientError as e:
                    logging.error(f"Errore durante la richiesta dell'intervallo {article_range}: {e}")

                # Opzionale: gestire il rate limit se necessario
                # await asyncio.sleep(2)

        if not save:
            return results

    async def process_additional_files(self, additional_files: List[str]):
        """
        Processa una lista di file aggiuntivi.

        :param additional_files: Lista di percorsi di file da processare.
        """
        for file_path in additional_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    logging.info(f"Processamento del file aggiuntivo: {file_path}")
                    await self.process_text(content)
                    logging.info(f"Completato il processamento del file: {file_path}")
                except Exception as e:
                    logging.error(f"Errore durante la lettura o il processamento del file {file_path}: {e}")
            else:
                logging.warning(f"File aggiuntivo non trovato: {file_path}")

    async def main_process(
        self,
        additional_files: List[str],
        act_details: List[dict]
    ):
        """
        Funzione principale per processare file aggiuntivi e articoli.

        :param additional_files: Lista di percorsi di file aggiuntivi.
        :param act_details: Lista di dizionari con dettagli sugli atti da processare.
        """
        try:
            # Processa i file aggiuntivi
            if additional_files:
                logging.info("Inizio processamento dei file aggiuntivi.")
                await self.process_additional_files(additional_files)
                logging.info("Completato il processamento dei file aggiuntivi.")

            # Processa gli articoli per ogni atto
            for act in act_details:
                logging.info(f"Inizio processamento per l'atto: {act['act_type']}.")
                await self.process_articles(
                    act_type=act.get("act_type", ""),
                    date=act.get("date", ""),
                    act_number=act.get("act_number", ""),
                    version=act.get("version", ""),
                    version_date=act.get("version_date", ""),
                    show_brocardi_info=act.get("show_brocardi_info", False),
                    articles=act.get("articles", []),
                    batch_size=act.get("batch_size", 50),
                    atomize=act.get("atomize", False),
                    save=act.get("save", True),
                    file_name=act.get("file_name", "output.txt")
                )
                logging.info(f"Completato il processamento per l'atto: {act['act_type']}.")

            # Salva i progressi
            self.base_rag.save_progress()
            logging.info("Tutti i processamenti sono stati completati con successo.")

        except Exception as e:
            logging.error(f"Si è verificato un errore nel processo principale: {e}")
            raise

    async def _wait_for_rate_limit(self) -> None:
        """
        Attende per rispettare i limiti di rate delle API.
        """
        now = time.time()
        elapsed_since_last_request = now - self.last_request_time
        elapsed_since_minute_start = now - self.minute_start_time

        # Reset dei contatori se è passato un minuto
        if elapsed_since_minute_start > 60:
            self.minute_start_time = now
            self.tokens_used_in_current_minute = 0
            logging.debug("Reset del conteggio dei token per il nuovo minuto.")

        # Attesa per rispettare il limite di richieste
        if elapsed_since_last_request < self.min_interval:
            sleep_time = self.min_interval - elapsed_since_last_request
            logging.debug(f"Attesa di {sleep_time:.2f} secondi per rispettare il limite di richieste.")
            await asyncio.sleep(sleep_time)

        self.last_request_time = time.time()

        # Attesa se il limite di token per minuto è stato raggiunto
        if self.tokens_used_in_current_minute >= self.tokens_per_minute:
            sleep_time = 60 - elapsed_since_minute_start
            logging.debug(f"Attesa di {sleep_time:.2f} secondi per rispettare il limite di token.")
            await asyncio.sleep(sleep_time)
            self.minute_start_time = time.time()
            self.tokens_used_in_current_minute = 0

    def _update_token_usage(self, tokens_used: int) -> None:
        """
        Aggiorna il conteggio dei token utilizzati.

        :param tokens_used: Numero di token utilizzati.
        """
        self.tokens_used_in_current_minute += tokens_used
        logging.debug(f"Token totali utilizzati nel minuto corrente: {self.tokens_used_in_current_minute}")

    async def _handle_rate_limit_error(self) -> None:
        """
        Gestisce gli errori di rate limit attendendo il tempo necessario prima di riprovare.
        """
        logging.warning("Rate limit raggiunto. Attesa di 60 secondi prima di riprovare.")
        await asyncio.sleep(60)
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Suddivide il testo in chunk di dimensioni gestibili basate sul numero massimo di token.

        :param text: Testo da suddividere.
        :return: Lista di chunk di testo.
        """
        tokens = self.encoding.encode(text)
        num_tokens = len(tokens)
        logging.debug(f"Suddivisione del testo di {num_tokens} token in chunk.")
        if num_tokens == 0:
            logging.warning("Testo vuoto fornito a split_text_into_chunks.")
            return []
        return [
            self.encoding.decode(tokens[i:i + self.max_chunk_tokens])
            for i in range(0, num_tokens, self.max_chunk_tokens)
        ]

    def create_article_ranges(self, articles: Union[List[int], Tuple[int, int]], batch_size: int) -> List[str]:
        if isinstance(articles, list):
            return [str(article) for article in articles]
        elif isinstance(articles, tuple):
            start, end = articles
            return [f"{i}-{min(i + batch_size - 1, end)}" for i in range(start, end + 1, batch_size)]
        else:
            raise ValueError("Il parametro 'articles' deve essere una lista di numeri di articoli o una tupla di range.")

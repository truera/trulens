import csv
import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional, Tuple
import zipfile

import pandas as pd
import requests
from tqdm.autonotebook import tqdm
from trulens.core import session as core_session

logger = logging.getLogger(__name__)


def download_and_unzip(url: str, out_dir: str, chunk_size: int = 1024) -> str:
    def download_url(url: str, save_path: str, chunk_size: int = 1024):
        """Download url with progress bar using tqdm
        https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

        Args:
            url (str): downloadable url
            save_path (str): local path to save the downloaded file
            chunk_size (int, optional): chunking of files. Defaults to 1024.
        """
        r = requests.get(url, stream=True)
        total = int(r.headers.get("Content-Length", 0))
        with open(save_path, "wb") as fd, tqdm(
            desc=save_path,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar:
            for data in r.iter_content(chunk_size=chunk_size):
                size = fd.write(data)
                bar.update(size)

    def unzip(zip_file: str, out_dir: str):
        zip_ = zipfile.ZipFile(zip_file, "r")
        zip_.extractall(path=out_dir)
        zip_.close()

    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)

    if not os.path.isfile(zip_file):
        logger.info("Downloading {} ...".format(dataset))
        download_url(url, zip_file, chunk_size)

    if not os.path.isdir(zip_file.replace(".zip", "")):
        logger.info("Unzipping {} ...".format(dataset))
        unzip(zip_file, out_dir)

    return os.path.join(out_dir, dataset.replace(".zip", ""))


class TruBEIRDataLoader:
    def __init__(
        self,
        data_folder: str,
        dataset_name: str,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
    ):
        """
        A utility class to load BEIR datasets into TruLens. Similar to https://github.com/beir-cellar/beir but with slightly
        more efficient implementation of processing and loading of the datasets using generators.

        Args:
            data_folder (str): local / remote path to store the downloaded dataset.
            dataset_name (str): Name of the dataset to be loaded. Must be one of https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/
            corpus_file (str, optional): file name of the corpus. Defaults to "corpus.jsonl".
            query_file (str, optional): file name of all the queries. Defaults to "queries.jsonl".
            qrels_folder (str, optional): folder name of qrels (relevance annotation). Defaults to "qrels".
            qrels_file (str, optional): file name of qrels (relevance annotation). Defaults to "".
        """
        self.qrels = {}

        self.dataset_name = dataset_name
        self.data_folder = data_folder

        self.corpus_file = (
            os.path.join(self.data_folder, dataset_name, corpus_file)
            if self.data_folder
            else corpus_file
        )
        self.query_file = (
            os.path.join(self.data_folder, dataset_name, query_file)
            if self.data_folder
            else query_file
        )
        self.qrels_folder = os.path.join(
            self.data_folder, dataset_name, qrels_folder
        )

        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(
                f"File {fIn} not present! Please provide accurate file."
            )

        if not fIn.endswith(ext):
            raise ValueError(f"File {fIn} must be present with extension {ext}")

    def _load_qrels(self):
        reader = csv.reader(
            open(self.qrels_file, encoding="utf-8"),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        next(reader)

        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score

    def _load_generators(
        self, split="test"
    ) -> Tuple[
        Generator[Dict[str, Dict[str, str]], None, None],
        Generator[Dict[str, Tuple[str, str]], None, None],
    ]:
        """
        Load corpus, queries, and qrels as generators.

        Args:
            split (str, optional): Dataset split to load. Defaults to "test".

        Returns:
            Tuple of generators for corpus, queries, and qrels.
        """
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        corpus_gen = self._corpus_generator()

        self._load_qrels()
        queries_gen = self._queries_generator()

        return corpus_gen, queries_gen

    def _corpus_generator(
        self,
    ) -> Generator[Dict[str, Dict[str, str]], None, None]:
        """
        Generator to load corpus data incrementally.
        """
        self.check(fIn=self.corpus_file, ext="jsonl")

        with open(self.corpus_file, encoding="utf8") as fIn:
            for line in fIn:
                line = json.loads(line)
                yield {
                    line.get("_id"): {
                        "text": line.get("text"),
                        "title": line.get("title"),
                    }
                }

    def _queries_generator(
        self,
    ) -> Generator[Dict[str, Tuple[str, str]], None, None]:
        """
        Generator to load queries incrementally.
        """
        self.check(fIn=self.query_file, ext="jsonl")

        with open(self.query_file, encoding="utf8") as fIn:
            for line in fIn:
                line = json.loads(line)
                if line.get("_id") in self.qrels:
                    yield {
                        line.get("_id"): (
                            line.get("text"),
                            line.get("metadata").get("answer"),
                        )
                    }

    def _process_dataset(
        self, split="test", chunk_size=None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Common method to process the BEIR dataset into entries.
        This method handles downloading, loading, and processing the dataset.

        Args:
            split (str, optional): Dataset split to load. Defaults to "test".
            chunk_size (int, optional): Number of records to process in each chunk. Defaults to None.

        Yields:
            List[Dict[str, Any]]: List of dataset entries (a chunk if chunk_size is specified).
        """

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"

        logger.info(f"Downloading {self.dataset_name} dataset from {url}")
        download_and_unzip(url, self.data_folder)

        corpus_gen, queries_gen = self._load_generators(split=split)

        dataset_entries = []

        corpus_dict = {}
        for corpus in list(
            corpus_gen
        ):  # TODO: (Daniel) this is still going to be a memory hog for large datasets, but if we don't
            # load the entire corpus into memory, we'll have to re-read the entire corpus using generator for each query
            corpus_dict.update(corpus)

        # Iterate over queries generator and process entries
        for i, query in enumerate(queries_gen, 1):
            for query_id, (query_text, answer) in query.items():
                doc_to_rel = self.qrels.get(query_id, {})

                expected_chunks = []

                for corpus_id in doc_to_rel.keys():
                    if corpus_id in corpus_dict:
                        corpus_entry = corpus_dict[corpus_id]
                        expected_chunks.append({
                            "text": corpus_entry["text"],
                            "title": corpus_entry.get("title"),
                            "expected_score": doc_to_rel.get(
                                corpus_id, 0
                            ),  # Default relevance score to 0 if not found
                        })

                dataset_entries.append({
                    "query_id": query_id,
                    "query": query_text,
                    "expected_response": answer,  # expected response can be also be empty for IR datasets
                    "expected_chunks": expected_chunks,
                    "meta": {"source": self.dataset_name},
                })

                # Yield the chunk if chunk_size is specified
                if chunk_size and i % chunk_size == 0:
                    yield dataset_entries
                    dataset_entries = []

        if dataset_entries:
            yield dataset_entries

    def load_dataset_to_df(
        self,
        split="test",
        download=True,
    ) -> pd.DataFrame:
        """
        load BEIR dataset into dataframe with pre-processed fields to match expected TruLens schemas.
        Note this method loads the entire dataset into memory at once.
        Args:
            split (str, optional): Defaults to "test".
            download (bool, optional): If False, remove the downloaded dataset file after processing. Defaults to True.
        Returns:
            pd.DataFrame: DataFrame with the BEIR dataset
        """
        dataset_entries = []
        for chunk in self._process_dataset(split=split):
            dataset_entries.extend(chunk)

        if not download:
            logger.info(f"Cleaning up downloaded {self.dataset_name} dataset")
            os.system(
                f"rm -rf {os.path.join(self.data_folder, self.dataset_name)}"
            )
            os.system(
                f"rm -rf {os.path.join(self.data_folder, self.dataset_name)}.zip"
            )

        return pd.DataFrame(dataset_entries)

    def persist_dataset(
        self,
        session: core_session.TruSession,
        dataset_name: str,
        dataset_metadata: Optional[Dict[str, Any]] = None,
        split="test",
        download=True,
        chunk_size=1000,
    ):
        """Persist BEIR dataset into DB with pre-processed fields to match expected TruLens schemas.

        Note this method handle chunking of the dataset to avoid loading the entire dataset into memory at once by default.

        Args:
            split: Defaults to "test".

            session: TruSession instance to persist the dataset.

            dataset_name: Name of the dataset to be persisted - Note this can
              be different from the standardized BEIR dataset names.

            dataset_metadata: Metadata for the dataset.

            download: If False, remove the downloaded dataset file after
              processing. Defaults to True.

        Returns:
            DataFrame with the BEIR dataset

        """
        for chunk in self._process_dataset(split=split, chunk_size=chunk_size):
            df_chunk = pd.DataFrame(chunk)
            session.add_ground_truth_to_dataset(
                dataset_name=dataset_name,
                ground_truth_df=df_chunk,
                dataset_metadata=dataset_metadata,
            )

        if not download:
            logger.info(f"Cleaning up downloaded {self.dataset_name} dataset")
            os.system(
                f"rm -rf {os.path.join(self.data_folder, self.dataset_name)}"
            )
            os.system(
                f"rm -rf {os.path.join(self.data_folder, self.dataset_name)}.zip"
            )

        logger.info(
            f"Finished processing dataset {self.dataset_name} in chunks."
        )

import csv
import json
import logging
import os
from typing import Dict, Tuple
import zipfile

import pandas as pd
import requests
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


BEIR_DATASET_NAMES = [
    "msmarco",
    "trec-covid",
    "nfcorpus",
    "nq",
    "hotpotqa",
    "fiqa",
    "arguana",
    "webis-touche2020",
    "cqadupstack",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact",
]


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
        prefix: str = "",
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if dataset_name not in BEIR_DATASET_NAMES:
            raise ValueError(
                f"Unknown dataset name: {dataset_name}. Must be one of {BEIR_DATASET_NAMES}"
            )
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

        if prefix:
            self.query_file = prefix + "-" + self.query_file
            self.qrels_folder = prefix + "-" + self.qrels_folder

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(
                f"File {fIn} not present! Please provide accurate file."
            )

        if not fIn.endswith(ext):
            raise ValueError(f"File {fIn} must be present with extension {ext}")

    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    def _load_corpus(self):
        num_lines = sum(1 for i in open(self.corpus_file, "rb"))
        with open(self.corpus_file, encoding="utf8") as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }

    def _load_queries(self):
        with open(self.query_file, encoding="utf8") as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")

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

    def _load(
        self, split="test"
    ) -> Tuple[
        Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]
    ]:
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info(
                "Loaded %d %s Documents.", len(self.corpus), split.upper()
            )
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info(
                "Loaded %d %s Queries.", len(self.queries), split.upper()
            )
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load_dataset_to_df(self, split="test") -> pd.DataFrame:
        """
        load BEIR dataset into dataframe with pre-processed fields to match expected TruLens schemas.

        Args:
            split (str, optional): Defaults to "test".

        Returns:
            pd.DataFrame: DataFrame with the BEIR dataset
        """

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"

        logger.info(f"Downloading {self.dataset_name} dataset from {url}")
        download_and_unzip(url, self.data_folder)

        corpus, queries, qrels = self._load(split=split)

        dataset_entries = []
        # Iterate over the queries generator and yield the dataset entries
        for query_id, query_text in queries.items():
            doc_to_rel = qrels.get(query_id, {})

            # Fetch the relevant documents lazily
            expected_chunks = []
            for corpus_id, corpus_entry in corpus.items():
                if corpus_id in doc_to_rel:
                    expected_chunks.append({
                        "text": corpus_entry["text"],
                        "title": corpus_entry.get("title"),
                        "expected_score": doc_to_rel.get(corpus_id),
                    })
                    doc_to_rel.pop(corpus_id)
                    if not doc_to_rel:
                        break

            dataset_entries.append({
                "query_id": query_id,
                "query": query_text,
                "expected_response": None,  # expected response can be empty for IR datasets
                "expected_chunks": expected_chunks,
                "meta": {"source": "BEIR", "domain": "IR"},
            })
        return pd.DataFrame(dataset_entries)

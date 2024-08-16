import csv
import json
import logging
import os
from typing import Generator, Tuple
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
        data_folder: str = "",
        prefix: str = "",
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
    ):
        self.corpus_file = (
            os.path.join(data_folder, corpus_file)
            if data_folder
            else corpus_file
        )
        self.query_file = (
            os.path.join(data_folder, query_file) if data_folder else query_file
        )
        self.qrels_folder = os.path.join(data_folder, qrels_folder)
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

    def _load_generators(
        self, split="test"
    ) -> Tuple[Generator, Generator, Generator]:
        """
        Lazily loads the corpus, queries, and qrels in a memory-efficient way.
        Returns generators instead of loading everything into memory.
        """

        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        corpus_generator = self._load_corpus_gen()
        queries_generator = self._load_queries_gen()
        qrels_generator = self._load_qrels_gen()

        return corpus_generator, queries_generator, qrels_generator

    def _load_corpus_gen(self) -> Generator:
        num_lines = sum(1 for i in open(self.corpus_file, "rb"))
        with open(self.corpus_file, encoding="utf8") as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                yield {
                    "_id": line.get("_id"),
                    "text": line.get("text"),
                    "title": line.get("title"),
                }

    def _load_queries_gen(self) -> Generator:
        with open(self.query_file, encoding="utf8") as fIn:
            for line in fIn:
                line = json.loads(line)
                yield {
                    "_id": line.get("_id"),
                    "text": line.get("text"),
                }

    def _load_qrels_gen(self) -> Generator:
        reader = csv.reader(
            open(self.qrels_file, encoding="utf-8"),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        next(reader)  # Skip header if present
        for row in reader:
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            yield {
                "query_id": query_id,
                "corpus_id": corpus_id,
                "score": score,
            }


def _get_beir_dataset_gen(dataset_name, data_path, split="test") -> Generator:
    """
    Get generators for BEIR dataset entries with pre-processed fields to match expected TruLens schemas.

    Args:
        dataset_name: Name of the BEIR dataset to load.
        data_path: Path where the dataset should be downloaded or is stored.

    Returns:
        Iterator over dataset entries.
    """
    if dataset_name not in BEIR_DATASET_NAMES:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. Must be one of {BEIR_DATASET_NAMES}"
        )

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = os.path.join(data_path, dataset_name)

    if not os.path.exists(out_dir):
        logger.info(f"Downloading {dataset_name} dataset to {out_dir}")
        download_and_unzip(url, data_path)

    corpus_gen, queries_gen, qrels_gen = TruBEIRDataLoader(
        data_folder=out_dir
    )._load_generators(split=split)
    # Convert the qrels generator to a dictionary to allow lookups
    qrels = {qrel["query_id"]: {} for qrel in qrels_gen}
    for qrel in qrels_gen:
        qrels[qrel["query_id"]][qrel["corpus_id"]] = qrel["score"]

    # Iterate over the queries generator and yield the dataset entries
    for query in queries_gen:
        query_id = query["_id"]
        query_text = query["text"]
        doc_to_rel = qrels.get(query_id, {})

        # Fetch the relevant documents lazily
        expected_chunks = []
        for corpus_entry in corpus_gen:
            if corpus_entry["_id"] in doc_to_rel:
                expected_chunks.append(
                    {
                        "text": corpus_entry["text"],
                        "title": corpus_entry.get("title"),
                    }
                )

        yield {
            "query_id": query_id,
            "query": query_text,
            "expected_response": qrels.get(query_id, None),
            "expected_chunks": expected_chunks,
            "dataset_id": dataset_name,
            "metadata": {"source": "BEIR", "dataset": dataset_name},
        }


# TODO: MOVE THIS TO tru.py
# def persist_beir_dataset(
#     dataset_name, data_path, split="test", chunk_size=1000
# ) -> pd.DataFrame:
#     """
#     Load BEIR dataset into a DataFrame in chunks.

#     Args:
#         dataset_name: Name of the BEIR dataset to load.
#         data_path: Path where the dataset should be downloaded or is stored.
#         split: Dataset split to load (e.g., "train", "test", "dev").
#         chunk_size: Number of records to process in each chunk.

#     Returns:
#         A pandas DataFrame containing the dataset entries.
#     """
#     dataset_gen = _get_beir_dataset_gen(dataset_name, data_path, split=split)

#     df = pd.DataFrame()

#     chunk = []
#     for idx, entry in enumerate(dataset_gen):
#         chunk.append(entry)

#         if (idx + 1) % chunk_size == 0:
#             df = pd.concat([df, pd.DataFrame(chunk)], ignore_index=True)
#             # TODO: write to DB
#             chunk = []  # Reset chunk

#     if chunk:
#         # TODO: write to DB
#         pass


def load_beir_dataset_df(dataset_name, data_path, split="test") -> pd.DataFrame:
    dataset_gen = _get_beir_dataset_gen(dataset_name, data_path, split=split)
    dataset_list = list(dataset_gen)
    df = pd.DataFrame(dataset_list)
    return df


# TODO: generalize to any dataset (user-provided dataframes)

from typing import Dict, Tuple, Union

import numpy as np
from trulens.core.utils.imports import REQUIREMENT_LLAMA
from trulens.core.utils.imports import REQUIREMENT_SKLEARN
from trulens.core.utils.imports import OptionalImports
from trulens.core.utils.pyschema import WithClassInfo
from trulens.core.utils.serial import SerialModel

with OptionalImports(messages=REQUIREMENT_SKLEARN):
    pass

with OptionalImports(messages=REQUIREMENT_LLAMA):
    from llama_index.core.base.embeddings.base import BaseEmbedding


class Embeddings(WithClassInfo, SerialModel):
    """Embedding related feedback function implementations."""

    _embed_model: BaseEmbedding

    def __init__(self, embed_model: BaseEmbedding):
        """Instantiates embeddings for feedback functions.
        !!! example

            Below is just one example. Embedders from llama-index are supported:
            https://docs.llamaindex.ai/en/latest/module_guides/models/embeddings/

            ```python
            from llama_index.embeddings.openai import OpenAIEmbedding
            from trulens_eval.feedback.embeddings import Embeddings

            embed_model = OpenAIEmbedding()

            f_embed = Embedding(embed_model=embed_model)
            ```

        Args:
            embed_model BaseEmbedding: Supports embedders from llama-index: https://docs.llamaindex.ai/en/latest/module_guides/models/embeddings/
        """
        super().__init__()
        self._embed_model = embed_model

    def cosine_distance(
        self, query: str, document: str
    ) -> Union[float, Tuple[float, Dict[str, str]]]:
        """
        Runs cosine distance on the query and document embeddings

        Example:

            Below is just one example. Embedders from llama-index are supported:
            https://docs.llamaindex.ai/en/latest/module_guides/models/embeddings/


            ```python
            from llama_index.embeddings.openai import OpenAIEmbedding
            from trulens_eval.feedback.embeddings import Embeddings

            embed_model = OpenAIEmbedding()

            # Create the feedback function
            f_embed = feedback.Embeddings(embed_model=embed_model)
            f_embed_dist = feedback.Feedback(f_embed.cosine_distance)\
                .on_input_output()
            ```

        Args:
            query (str): A text prompt to a vector DB.
            document (str): The document returned from the vector DB.

        Returns:
            - float: the embedding vector distance
        """
        import sklearn

        query_embed = np.asarray(
            self._embed_model.get_query_embedding(query)
        ).reshape(
            1, -1
        )  # sklearn expects 2d array (first dimension number of samples)
        document_embed = np.asarray(
            self._embed_model.get_text_embedding(document)
        ).reshape(
            1, -1
        )  # sklearn expects 2d array (first dimension number of samples)

        return sklearn.metrics.pairwise.cosine_distances(
            query_embed, document_embed
        )[
            0
        ][
            0
        ]  # final results will be dimensions (sample query x sample doc) === (1,1)

    def manhattan_distance(
        self, query: str, document: str
    ) -> Union[float, Tuple[float, Dict[str, str]]]:
        """
        Runs L1 distance on the query and document embeddings

        Example:

            Below is just one example. Embedders from llama-index are supported:
            https://docs.llamaindex.ai/en/latest/module_guides/models/embeddings/

            ```python
            from llama_index.embeddings.openai import OpenAIEmbedding
            from trulens_eval.feedback.embeddings import Embeddings

            embed_model = OpenAIEmbedding()

            # Create the feedback function
            f_embed = feedback.Embeddings(embed_model=embed_model)
            f_embed_dist = feedback.Feedback(f_embed.manhattan_distance)\
                .on_input_output()
            ```

        Args:
            query (str): A text prompt to a vector DB.
            document (str): The document returned from the vector DB.

        Returns:
            - float: the embedding vector distance
        """
        import sklearn

        query_embed = np.asarray(
            self._embed_model.get_query_embedding(query)
        ).reshape(
            1, -1
        )  # sklearn expects 2d array (first dimension number of samples)
        document_embed = np.asarray(
            self._embed_model.get_text_embedding(document)
        ).reshape(
            1, -1
        )  # sklearn expects 2d array (first dimension number of samples)

        return sklearn.metrics.pairwise.manhattan_distances(
            query_embed, document_embed
        )[
            0
        ][
            0
        ]  # final results will be dimensions (sample query x sample doc) === (1,1)

    def euclidean_distance(
        self, query: str, document: str
    ) -> Union[float, Tuple[float, Dict[str, str]]]:
        """
        Runs L2 distance on the query and document embeddings

        Example:

            Below is just one example. Embedders from llama-index are supported:
            https://docs.llamaindex.ai/en/latest/module_guides/models/embeddings/

            ```python
            from llama_index.embeddings.openai import OpenAIEmbedding
            from trulens_eval.feedback.embeddings import Embeddings

            embed_model = OpenAIEmbedding()

            # Create the feedback function
            f_embed = feedback.Embeddings(embed_model=embed_model)
            f_embed_dist = feedback.Feedback(f_embed.euclidean_distance)\
                .on_input_output()
            ```

        Args:
            query (str): A text prompt to a vector DB.
            document (str): The document returned from the vector DB.

        Returns:
            - float: the embedding vector distance
        """
        import sklearn

        query_embed = np.asarray(
            self._embed_model.get_query_embedding(query)
        ).reshape(
            1, -1
        )  # sklearn expects 2d array (first dimension number of samples)
        document_embed = np.asarray(
            self._embed_model.get_text_embedding(document)
        ).reshape(
            1, -1
        )  # sklearn expects 2d array (first dimension number of samples)

        return sklearn.metrics.pairwise.euclidean_distances(
            query_embed, document_embed
        )[
            0
        ][
            0
        ]  # final results will be dimensions (sample query x sample doc) === (1,1)

from typing import Dict, Tuple, Union

import numpy as np
import pydantic
from trulens.core.utils.imports import REQUIREMENT_SKLEARN
from trulens.core.utils.imports import OptionalImports
from trulens.core.utils.imports import format_import_errors
from trulens.core.utils.pyschema import WithClassInfo
from trulens.core.utils.serial import SerialModel

with OptionalImports(REQUIREMENT_SKLEARN):
    import sklearn.metrics

with OptionalImports(
    messages=format_import_errors(
        "llama-index", purpose="using llama-index embedding models"
    )
):
    from llama_index.core.base.embeddings.base import BaseEmbedding


class Embeddings(WithClassInfo, SerialModel):
    """Embedding related feedback function implementations.

    !!!  Warning
        This class cannot presently be used in evaluating feedback in deferred
        mode.
    """

    embed_model: BaseEmbedding
    """llama_index-based embedding."""
    # NOTE: As of current llama_index, this is a pydantic.v1.BaseModel which we
    # cannot normally include inside a v2 (what we use) model. We need to
    # override the validator because of this otherwise we get weird errors. We
    # also cannot use automatic serialization for the same reason.

    @pydantic.field_validator("embed_model", mode="plain")
    @classmethod
    def _validate_embed_model(cls, v):
        if not isinstance(v, BaseEmbedding):
            raise ValueError(
                "embed_model must be an instance of llama_index.core.base.embeddings.base.BaseEmbedding"
            )
        return v

    def __init__(self, embed_model: BaseEmbedding):
        """Instantiates embeddings for feedback functions.
        !!! example

            Below is just one example. Embedders from llama-index are supported:
            https://docs.llamaindex.ai/en/latest/module_guides/models/embeddings/

            ```python
            from llama_index.embeddings.openai import OpenAIEmbedding
            from trulens.feedback.embeddings import Embeddings

            embed_model = OpenAIEmbedding()

            f_embed = Embedding(embed_model=embed_model)
            ```

        Args:
            embed_model: Supports embedders from llama-index:
                https://docs.llamaindex.ai/en/latest/module_guides/models/embeddings/
        """
        super().__init__(embed_model=embed_model)

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
            from trulens.feedback.embeddings import Embeddings

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

        query_embed = np.asarray(
            self.embed_model.get_query_embedding(query)
        ).reshape(
            1, -1
        )  # sklearn expects 2d array (first dimension number of samples)
        document_embed = np.asarray(
            self.embed_model.get_text_embedding(document)
        ).reshape(
            1, -1
        )  # sklearn expects 2d array (first dimension number of samples)

        # final results will be dimensions (sample query x sample doc) === (1,1)
        return sklearn.metrics.pairwise.cosine_distances(
            query_embed, document_embed
        )[0][0]

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
            from trulens.feedback.embeddings import Embeddings

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

        query_embed = np.asarray(
            self.embed_model.get_query_embedding(query)
        ).reshape(
            1, -1
        )  # sklearn expects 2d array (first dimension number of samples)
        document_embed = np.asarray(
            self.embed_model.get_text_embedding(document)
        ).reshape(
            1, -1
        )  # sklearn expects 2d array (first dimension number of samples)

        # final results will be dimensions (sample query x sample doc) === (1,1)
        return sklearn.metrics.pairwise.manhattan_distances(
            query_embed, document_embed
        )[0][0]

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
            from trulens.feedback.embeddings import Embeddings

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

        query_embed = np.asarray(
            self.embed_model.get_query_embedding(query)
        ).reshape(
            1, -1
        )  # sklearn expects 2d array (first dimension number of samples)
        document_embed = np.asarray(
            self.embed_model.get_text_embedding(document)
        ).reshape(
            1, -1
        )  # sklearn expects 2d array (first dimension number of samples)

        # final results will be dimensions (sample query x sample doc) === (1,1)
        return sklearn.metrics.pairwise.euclidean_distances(
            query_embed, document_embed
        )[0][0]

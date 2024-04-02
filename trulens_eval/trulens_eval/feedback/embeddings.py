from typing import Dict, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from trulens_eval.utils.imports import OptionalImports
from trulens_eval.utils.imports import REQUIREMENT_LLAMA
from trulens_eval.utils.imports import REQUIREMENT_SKLEARN
from trulens_eval.utils.pyschema import WithClassInfo
from trulens_eval.utils.serial import SerialModel

with OptionalImports(messages=REQUIREMENT_SKLEARN):
    import sklearn

with OptionalImports(messages=REQUIREMENT_LLAMA):
    from llama_index.legacy import ServiceContext


class Embeddings(WithClassInfo, SerialModel):
    """Embedding related feedback function implementations.
    """
    _embed_model: 'Embedder' = PrivateAttr()

    def __init__(self, embed_model: 'Embedder' = None):
        """Instantiates embeddings for feedback functions. 
        ```
        f_embed = feedback.Embeddings(embed_model=embed_model)
        ```

        Args:
            embed_model ('Embedder'): Supported embedders taken from llama-index: https://gpt-index.readthedocs.io/en/latest/core_modules/model_modules/embeddings/root.html
        """

        service_context = ServiceContext.from_defaults(embed_model=embed_model)
        self._embed_model = service_context.embed_model
        super().__init__()

    def cosine_distance(
        self, query: str, document: str
    ) -> Union[float, Tuple[float, Dict[str, str]]]:
        """
        Runs cosine distance on the query and document embeddings

        !!! example
    
            Below is just one example. See supported embedders:
            https://gpt-index.readthedocs.io/en/latest/core_modules/model_modules/embeddings/root.html
            from langchain.embeddings.openai import OpenAIEmbeddings

            ```python
            model_name = 'text-embedding-ada-002'

            embed_model = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=OPENAI_API_KEY
            )

            # Create the feedback function
            f_embed = feedback.Embeddings(embed_model=embed_model)
            f_embed_dist = feedback.Feedback(f_embed.cosine_distance)\
                .on_input()\
                .on(Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content)
            ```

            The `on(...)` selector can be changed. See [Feedback Function Guide
            :
            Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)

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
        )[0][
            0
        ]  # final results will be dimensions (sample query x sample doc) === (1,1)

    def manhattan_distance(
        self, query: str, document: str
    ) -> Union[float, Tuple[float, Dict[str, str]]]:
        """
        Runs L1 distance on the query and document embeddings

        !!! example
    
            Below is just one example. See supported embedders:
            https://gpt-index.readthedocs.io/en/latest/core_modules/model_modules/embeddings/root.html
            from langchain.embeddings.openai import OpenAIEmbeddings

            ```python
            model_name = 'text-embedding-ada-002'

            embed_model = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=OPENAI_API_KEY
            )

            # Create the feedback function
            f_embed = feedback.Embeddings(embed_model=embed_model)
            f_embed_dist = feedback.Feedback(f_embed.manhattan_distance)\
                .on_input()\
                .on(Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content)
            ```

            The `on(...)` selector can be changed. See [Feedback Function Guide
            :
            Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)

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
        )[0][
            0
        ]  # final results will be dimensions (sample query x sample doc) === (1,1)

    def euclidean_distance(
        self, query: str, document: str
    ) -> Union[float, Tuple[float, Dict[str, str]]]:
        """
        Runs L2 distance on the query and document embeddings

        !!! example
    
            Below is just one example. See supported embedders:
            https://gpt-index.readthedocs.io/en/latest/core_modules/model_modules/embeddings/root.html
            from langchain.embeddings.openai import OpenAIEmbeddings
            
            ```python
            model_name = 'text-embedding-ada-002'

            embed_model = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=OPENAI_API_KEY
            )

            # Create the feedback function
            f_embed = feedback.Embeddings(embed_model=embed_model)
            f_embed_dist = feedback.Feedback(f_embed.euclidean_distance)\
                .on_input()\
                .on(Select.Record.app.combine_documents_chain._call.args.inputs.input_documents[:].page_content)
            ```

            The `on(...)` selector can be changed. See [Feedback Function Guide
            :
            Selectors](https://www.trulens.org/trulens_eval/feedback_function_guide/#selector-details)

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
        )[0][
            0
        ]  # final results will be dimensions (sample query x sample doc) === (1,1)

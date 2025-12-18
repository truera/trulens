import json
import logging
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class NuggetizerProvider:
    """Provider for parsing responses into atomic nuggets."""

    def __init__(
        self, model: str = "gpt-5-nano", api_key: Optional[str] = None
    ):
        """
        Initialize nuggetizer provider.

        Args:
            model: OpenAI model to use for nugget extraction (default: gpt-5-nano)
            api_key: OpenAI API key (uses environment variable if not provided)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.cache = {}
        logger.info(f"Initialized NuggetizerProvider with model: {model}")

    def extract_nuggets(
        self,
        context_text: str,
        query_text: str,
        max_nuggets: int = 30,
        nuggets=[],
    ) -> List[str]:
        """
        Extract atomic nuggets from text using LLM.

        Args:
            context_text: context text to extract nuggets from
            query_text: a piece of query text to condition nugget extraction on.
            max_nuggets: Maximum number of nuggets to extract

        Returns:
            List of nuggets
        """
        prompt = f"""Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process).  Return only the final list of all nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. Ensure the updated nugget list has at most {max_nuggets} nuggets (can be less), keeping only the most vital ones. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information.
                Search Query: {query_text}
                Context:
                {context_text}
                Search Query: {query_text}
                Initial Nugget List: {nuggets}
                Initial Nugget List Length: {len(nuggets)}

                Only update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions). List in the form ["a", "b", ...] and a and b are strings with no mention of ".
                Updated Nugget List:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()
            return json.loads(content)

        except Exception as e:
            logger.error(f"Error extracting nuggets: {e}")
            return []

    def classify_nuggets(
        self,
        nuggets: List[str],
        query_text: str,
    ) -> List[str]:
        """
        Classify the importance of nuggets from text using LLM.

        Args:
            nuggets: a list of facts or statements which cannot be broken down further.
            query_text: a piece of query text to condition nugget extration on.

        Returns:
            List of nugget importance mapping the importance of inputted nuggets
        """
        prompt = f"""Based on the query, label each of the {len(nuggets)} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a "good" answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

                Search Query: {query_text}
                Nugget List: {[nugget for nugget in nuggets]}

                Only return the list of labels (List[str]). Do not explain.
                Labels:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()
            return json.loads(content)

        except Exception as e:
            logger.error(f"Error classifying nuggets: {e}")
            return []

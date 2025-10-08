import json
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from openai import OpenAI

from trulens.core.feedback import feedback as core_feedback
from trulens.core.utils import serial as serial_utils
from trulens.feedback import llm_provider

logger = logging.getLogger(__name__)


class NuggetizerProvider:
    """Provider for parsing responses into atomic nuggets."""
    
    def __init__(self, model: str = "gpt-5-nano", api_key: Optional[str] = None, system_prompt: str = "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query."):
        """
        Initialize nuggetizer provider.
        
        Args:
            model: OpenAI model to use for nugget extraction (default: gpt-5-nano)
            api_key: OpenAI API key (uses environment variable if not provided)
        """
        self.model = model
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.cache = {}
        self.system_prompt = system_prompt
        logger.info(f"Initialized NuggetizerProvider with model: {model}")
    
    
    def extract_nuggets(self, context_text: str, query_text: str, max_nuggets: int = 30, nuggets = []) -> List[str]:
        """
        Extract atomic nuggets from text using LLM.
        
        Args:
            context_text: context text to extract nuggets from
            query_text: a piece of query text to condition nugget extration on. 
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
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            return list(content)
            
                
        except Exception as e:
            logger.error(f"Error extracting nuggets: {e}")
            return []
    

    def classify_nuggets(self, nuggets: List[str], query_text: str,) -> List[str]:
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
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            return list(content)
            
                
        except Exception as e:
            logger.error(f"Error extracting nuggets: {e}")
            return []
    



class NuggetizedFeedback:
    """
    Wrapper that adds nuggetization on top of standard TruLens feedback functions.
    
    This class enables nugget-level evaluation without modifying TruLens core.
    """
    
    def __init__(
        self, 
        provider: llm_provider.LLMProvider,
        nuggetizer: Optional[NuggetizerProvider] = None
    ):
        """
        Initialize nuggetized feedback wrapper.
        
        Args:
            provider: Standard TruLens LLM provider for evaluation
            nuggetizer: Optional nuggetizer provider (creates default if not provided)
        """
        self.provider = provider
        self.nuggetizer = nuggetizer or NuggetizerProvider()
        logger.info(f"Initialized NuggetizedFeedback with provider: {provider.__class__.__name__}")
    
    def groundedness_nuggetized(
        self, 
        source: str, 
        statement: str
    ) -> Tuple[float, Dict]:
        """
        Evaluate groundedness at nugget level.
        
        Args:
            source: Source context
            statement: Statement to evaluate
            
        Returns:
            Tuple of (aggregate_score, metadata_dict)
        """
        # Extract nuggets from statement
        nuggets = self.nuggetizer.extract_nuggets(statement)
        logger.debug(f"Extracted {len(nuggets)} nuggets from statement")
        
        # Evaluate each nugget
        nugget_evaluations = []
        for nugget in nuggets:
            score, reasons = self.provider.groundedness_measure_with_cot_reasons(
                source=source,
                statement=nugget['text']
            )
            
            nugget_evaluations.append({
                'nugget': nugget,
                'score': score,
                'reasons': reasons
            })
        
        # Aggregate with importance weighting
        aggregate_score = self._aggregate_scores(
            nugget_evaluations, 
            score_key='score'
        )
        
        metadata = {
            'nugget_evaluations': nugget_evaluations,
            'total_nuggets': len(nuggets),
            'method': 'nuggetized'
        }
        
        return aggregate_score, metadata
    
    def relevance_nuggetized(
        self, 
        prompt: str, 
        response: str
    ) -> Tuple[float, Dict]:
        """
        Evaluate answer relevance at nugget level.
        
        Args:
            prompt: Question/prompt
            response: Answer/response to evaluate
            
        Returns:
            Tuple of (aggregate_score, metadata_dict)
        """
        # Extract nuggets from response
        nuggets = self.nuggetizer.extract_nuggets(response)
        logger.debug(f"Extracted {len(nuggets)} nuggets from response")
        
        # Evaluate each nugget
        nugget_evaluations = []
        for nugget in nuggets:
            score, reasons = self.provider.relevance_with_cot_reasons(
                prompt=prompt,
                response=nugget['text']
            )
            
            nugget_evaluations.append({
                'nugget': nugget,
                'score': score,
                'reasons': reasons
            })
        
        # Aggregate with importance weighting
        aggregate_score = self._aggregate_scores(
            nugget_evaluations,
            score_key='score'
        )
        
        metadata = {
            'nugget_evaluations': nugget_evaluations,
            'total_nuggets': len(nuggets),
            'method': 'nuggetized'
        }
        
        return aggregate_score, metadata
    
    def _aggregate_scores(
        self, 
        nugget_evaluations: List[Dict],
        score_key: str = 'score'
    ) -> float:
        """
        Aggregate nugget scores using importance weighting.
        
        Args:
            nugget_evaluations: List of nugget evaluation dictionaries
            score_key: Key for score in evaluation dict
            
        Returns:
            Weighted average score
        """
        if not nugget_evaluations:
            return 0.0
        
        total_importance = sum(
            eval['nugget']['importance'] 
            for eval in nugget_evaluations
        )
        
        if total_importance == 0:
            # Fallback to equal weighting
            return np.mean([eval[score_key] for eval in nugget_evaluations])
        
        weighted_sum = sum(
            eval[score_key] * eval['nugget']['importance']
            for eval in nugget_evaluations
        )
        
        return weighted_sum / total_importance


def nuggetized_rag_triad(
    provider: llm_provider.LLMProvider,
    question: Optional[serial_utils.Lens] = None,
    answer: Optional[serial_utils.Lens] = None,
    context: Optional[serial_utils.Lens] = None,
    nuggetizer: Optional[NuggetizerProvider] = None,
    use_nuggets: bool = True
) -> Dict[str, core_feedback.Feedback]:
    """
    Create a nuggetized RAG triad for granular evaluation.
    
    This function creates feedback functions that evaluate at the nugget level,
    providing more detailed insights than traditional evaluation.
    
    Args:
        provider: The provider to use for implementing the feedback functions
        question: Selector for the question part
        answer: Selector for the answer part
        context: Selector for the context part
        nuggetizer: Optional nuggetizer provider (creates default if not provided)
        use_nuggets: Whether to use nuggetization (True) or fall back to standard (False)
        
    Returns:
        Dictionary of feedback functions with nugget-level evaluation
    """
    
    if not use_nuggets:
        # Fall back to standard RAG triad
        from trulens.feedback.feedback import rag_triad
        return rag_triad(provider, question, answer, context)
    
    # Initialize nuggetized wrapper
    nugget_wrapper = NuggetizedFeedback(provider, nuggetizer)
    
    assert hasattr(
        provider, "relevance_with_cot_reasons"
    ), "Need a provider with the `relevance_with_cot_reasons` feedback function."
    assert hasattr(
        provider, "context_relevance_with_cot_reasons"
    ), "Need a provider with the `context_relevance_with_cot_reasons` feedback function."
    assert hasattr(
        provider, "groundedness_measure_with_cot_reasons"
    ), "Need a provider with the `groundedness_measure_with_cot_reasons` feedback function."
    
    are_complete: bool = True
    ret = {}
    
    # Create nuggetized feedback functions
    for f_imp, f_agg, arg1name, arg1lens, arg2name, arg2lens, f_name in [
        (
            nugget_wrapper.groundedness_nuggetized,
            None,  # No additional aggregation needed - already aggregated
            "source",
            context and context.collect(),
            "statement",
            answer,
            "Groundedness (Nuggetized)",
        ),
        (
            nugget_wrapper.relevance_nuggetized,
            None,  # No additional aggregation needed - already aggregated
            "prompt",
            question,
            "response",
            answer,
            "Answer Relevance (Nuggetized)",
        ),
        (
            # Context relevance doesn't need nuggetization
            provider.context_relevance_with_cot_reasons,
            np.mean,
            "question",
            question,
            "context",
            context,
            "Context Relevance",
        ),
    ]:
        f = core_feedback.Feedback(
            f_imp, if_exists=context, name=f_name
        )
        
        # Only aggregate if specified
        if f_agg is not None:
            f = f.aggregate(f_agg)
        
        if arg1lens is not None:
            f = f.on(**{arg1name: arg1lens})
        else:
            are_complete = False
        
        if arg2lens is not None:
            f = f.on(**{arg2name: arg2lens})
        else:
            are_complete = False
        
        ret[f.name] = f
    
    if not are_complete:
        logger.warning(
            "Some or all nuggetized RAG triad feedback functions do not have all their selectors set. "
            "This may be ok if they are to be used for colang actions."
        )
    
    return ret


def compare_traditional_vs_nuggetized(
    provider: llm_provider.LLMProvider,
    question: str,
    answer: str,
    context: str,
    nuggetizer: Optional[NuggetizerProvider] = None
) -> Dict:
    """
    Compare traditional vs nuggetized evaluation side-by-side.
    
    This utility function is useful for testing and demonstrating the
    differences between traditional and nuggetized evaluation.
    
    Args:
        provider: TruLens LLM provider
        question: Question text
        answer: Answer text
        context: Context text
        nuggetizer: Optional nuggetizer provider
        
    Returns:
        Dictionary with comparison results
    """
    # Traditional evaluation
    trad_groundedness, trad_g_reasons = provider.groundedness_measure_with_cot_reasons(
        source=context,
        statement=answer
    )
    trad_relevance, trad_r_reasons = provider.relevance_with_cot_reasons(
        prompt=question,
        response=answer
    )
    
    # Nuggetized evaluation
    nugget_wrapper = NuggetizedFeedback(provider, nuggetizer)
    nugget_groundedness, nugget_g_metadata = nugget_wrapper.groundedness_nuggetized(
        source=context,
        statement=answer
    )
    nugget_relevance, nugget_r_metadata = nugget_wrapper.relevance_nuggetized(
        prompt=question,
        response=answer
    )
    
    return {
        'traditional': {
            'groundedness': trad_groundedness,
            'groundedness_reasons': trad_g_reasons,
            'relevance': trad_relevance,
            'relevance_reasons': trad_r_reasons
        },
        'nuggetized': {
            'groundedness': nugget_groundedness,
            'groundedness_metadata': nugget_g_metadata,
            'relevance': nugget_relevance,
            'relevance_metadata': nugget_r_metadata
        },
        'comparison': {
            'groundedness_diff': nugget_groundedness - trad_groundedness,
            'relevance_diff': nugget_relevance - trad_relevance,
            'total_nuggets': nugget_g_metadata['total_nuggets']
        }
    }
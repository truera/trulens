# Nuggetized Feedback
Information nuggets were introduced long ago as part of an early [TREC Track](https://trec.nist.gov/pubs/trec12/papers/QA.OVERVIEW.pdf) focused on question answering. This track defined the category as: "An information nugget was defined as a fact for which the assessor could make a binary decision as to whether a response contained the nugget. At the end of this step, the assessor decided which nuggets were vital—nuggets that must appear in a definition for that definition to be good". In other words, an information nugget is a fact or statement which when presented with an answer, it is immediately visible if this "nugget" is present.

In the 2024 and beyond RAG tracks, nuggets became the default way of understanding response qualities of RAG systems as it could verify the presence of indivisible facts or statements and verify the presence. 

## Nuggetization Process. (Original)
As shown in [The Great Nugget Recall](https://arxiv.org/pdf/2504.15068) the nuggetization process is a simple multi step process which goes as follows:
1. First, given a piece of context and a user query or intent, extract the relevant nuggets. This runs iteratively until no new nuggets are created or a max threshold is met (30 in TREC RAG)
2. Given this set of nuggets, they are then scored for importance relative to query needs. The labels are either "vital" or "ok". "ok" is related but not needed while "vital" is, as implied, vital to the query. 
3. Once there are importance scores, the final step is to measure if nuggets are supported by the passage which they come from. Nuggets/context relationship can be "support" implying the nugget is fully supported by the passage, "not_support", and "partial_support", which imply what their names indicate.

## Nuggets in TruLens
True Lens is build around the notion of RAG triad which decomposes RAG answer quality into context relevance, groundeness, and answer relevance.Context relevance measures the relavance of the seleted context to the original query, groundeness measure how grounded in the context the answer is, and answer relevance measure how relevant the answer is to the question. This approach is simple and effective but lacks the expresivity of the nugget approach.

To make nuggets fit within the RAG triad we treat the process of nuggetization as a wrapper to the existing feedback process. Context relevance is unchanged so existing passages are evaluated on relevance with direct comparing the context passage with the query. Groundness and answer relevance are changed as they the existing RAG triad gets nuggetized information. This all means that Nuggetized RAG triad behaves as follows

1. First, given a piece of context and a user query or intent, extract the relevant nuggets. This runs iteratively until no new nuggets are created or a max threshold is met (30 in TREC RAG). This is done based on the full generated answer. 
2. Given this set of nuggets, they are then scored for importance relative to query needs. The labels are either "vital" or "ok". "ok" is related but not needed while "vital" is, as implied, vital to the query. 
3. Nuggets are passed to the existing RAG triad and evaluated independently. 
4. RAG triad results are combined and weighted based on nugget importance. 

This process is demonstrated by running the script below. Be sure to set up your OPEN_AI_API_KEY before proceeding. 



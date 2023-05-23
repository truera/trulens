import logging
import os
from pprint import PrettyPrinter
from typing import Callable, Dict, List, Set, Tuple

import numpy as np

# This needs to be before some others to make sure api keys are ready before
# relevant classes are loaded.
from trulens_eval.keys import *


# This is here so that import organizer does not move the keys import below this
# line.
_ = None

from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import Document
from langchain.vectorstores import Pinecone
from langchain.vectorstores.base import VectorStoreRetriever
import pinecone
from pydantic import Field
from slack_bolt import App
from slack_sdk import WebClient

from trulens_eval import Tru
from trulens_eval import tru_feedback
from trulens_eval.tru_chain import TruChain
from trulens_eval.tru_db import LocalSQLite
from trulens_eval.tru_db import Record
from trulens_eval.tru_feedback import Feedback
from trulens_eval.util import TP

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

pp = PrettyPrinter()

PORT = 3000
verb = False

# create a conversational chain with relevant models and vector store

# Pinecone configuration.
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)

# Cache of conversations. Keys are SlackAPI conversation ids (channel ids or
# otherwise) and values are TruChain to handle that conversation.
convos: Dict[str, TruChain] = dict()

# Keep track of timestamps of messages already handled. Sometimes the same
# message gets received more than once if there is a network hickup.
handled_ts: Set[Tuple[str, str]] = set()

# DB to save models and records.
tru = Tru()#LocalSQLite("trubot.sqlite"))

ident = lambda h: h

chain_ids = {
    0: "0/default",
    1: "1/lang_prompt",
    2: "2/relevance_prompt",
    3: "3/filtered_context",
    4: "4/filtered_context_and_lang_prompt"
}

# Construct feedback functions.

hugs = tru_feedback.Huggingface()
openai = tru_feedback.OpenAI()

# Language match between question/answer.
f_lang_match = Feedback(hugs.language_match).on(
    text1="prompt", text2="response"
)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on(
    prompt="input", response="output"
)

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai.qs_relevance).on(
    question="input",
    statement=Record.chain.combine_docs_chain._call.args.inputs.input_documents
).on_multiple(
    multiarg="statement", each_query=Record.page_content, agg=np.min
)

class WithFilterDocuments(VectorStoreRetriever):
    filter_func: Callable = Field(exclude=True)

    def __init__(self, filter_func: Callable, *args, **kwargs):
        super().__init__(filter_func=filter_func, *args, **kwargs)
        # self.filter_func = filter_func

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = super().get_relevant_documents(query)

        promises = []
        for doc in docs:
            promises.append(
                (doc, TP().promise(self.filter_func, query=query, doc=doc))
            )

        results = []
        for doc, promise in promises:
            results.append((doc, promise.get()))

        docs_filtered = map(lambda sr: sr[0], filter(lambda sr: sr[1], results))

        return list(docs_filtered)

    @staticmethod
    def of_vectorstoreretriever(retriever, filter_func: Callable):
        return WithFilterDocuments(filter_func=filter_func, **retriever.dict())

def filter_by_relevance(query, doc):
    return openai.qs_relevance(question=query, statement=doc.page_content) > 0.5

def get_or_make_chain(cid: str, selector: int = 0) -> TruChain:
    """
    Create a new chain for the given conversation id `cid` or return an existing
    one. Return the new or existing chain.
    """

    # NOTE(piotrm): Unsure about the thread safety of the various components so
    # making new ones for each conversation.

    if cid in convos:
        return convos[cid]

    if selector not in chain_ids:
        selector = 0

    chain_id = chain_ids[selector]

    pp.pprint(f"Starting a new conversation with {chain_id}.")

    # Embedding needed for Pinecone vector db.
    embedding = OpenAIEmbeddings(model='text-embedding-ada-002')  # 1536 dims
    docsearch = Pinecone.from_existing_index(
        index_name="llmdemo", embedding=embedding
    )

    retriever = docsearch.as_retriever()

    if "filtered" in chain_id:
        retriever = WithFilterDocuments.of_vectorstoreretriever(
            retriever=retriever, filter_func=filter_by_relevance
        )

    # LLM for completing prompts, and other tasks.
    llm = OpenAI(temperature=0, max_tokens=128)

    # Conversation memory.
    memory = ConversationSummaryBufferMemory(
        max_token_limit=650,
        llm=llm,
        memory_key="chat_history",
        output_key='answer'
    )

    # Conversational chain puts it all together.
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=verb,
        return_source_documents=True,
        memory=memory,
        get_chat_history=ident,
        max_tokens_limit=4096
    )

    # Need to copy these otherwise various chains will feature templates that
    # point to the same objects.
    chain.combine_docs_chain.llm_chain.prompt = \
        chain.combine_docs_chain.llm_chain.prompt.copy()
    chain.combine_docs_chain.document_prompt = \
        chain.combine_docs_chain.document_prompt.copy()

    if "lang" in chain_id:
        # Language mismatch fix:
        chain.combine_docs_chain.llm_chain.prompt.template = \
            "Use the following pieces of context to answer the question at the end " \
            "in the same language as the question. If you don't know the answer, " \
            "just say that you don't know, don't try to make up an answer.\n" \
            "\n" \
            "{context}\n" \
            "\n" \
            "Question: {question}\n" \
            "Helpful Answer: "

    elif "relevance" in chain_id:
        # Contexts fix

        # whitespace important in "Contexts! "
        chain.combine_docs_chain.llm_chain.prompt.template = \
            "Use only the relevant contexts to answer the question at the end " \
            ". Some pieces of context may not be relevant. If you don't know the answer, " \
            "just say that you don't know, don't try to make up an answer.\n" \
            "\n" \
            "Contexts: \n" \
            "{context}\n" \
            "\n" \
            "Question: {question}\n" \
            "Helpful Answer: "

        # "\t" important here:
        chain.combine_docs_chain.document_prompt.template = "\tContext: {page_content}"

    # Trulens instrumentation.
    tc = tru.Chain(
        chain=chain,
        chain_id=chain_id,
        feedbacks=[f_lang_match, f_qa_relevance, f_qs_relevance],
        feedback_mode="deferred"
    )

    convos[cid] = tc

    return tc


def get_answer(chain: TruChain, question: str) -> Tuple[str, str]:
    """
    Use the given `chain` to respond to `question`. Return the answer text and
    sources elaboration text.
    """

    # Pace our API usage. This is not perfect since the chain makes multiple api calls
    # internally.
    openai.endpoint.pace_me()

    outs = chain(dict(question=question))

    result = outs['answer']
    sources = outs['source_documents']

    result_sources = "Sources:\n"

    temp = set()

    for doc in sources:
        src = doc.metadata['source']
        if src not in temp:
            result_sources += " - " + doc.metadata['source']
            if 'page' in doc.metadata:
                result_sources += f" (page {int(doc.metadata['page'])})\n"
            else:
                result_sources += "\n"

            temp.add(src)

    return result, result_sources


def answer_message(client, body: dict, logger):
    """
    SlackAPI handler of message received.
    """

    pp.pprint(body)

    ts = body['event']['ts']
    user = body['event']['user']

    if (ts, user) in handled_ts:
        print(f"WARNING: I already handled message with ts={ts}, user={user} .")
        return
    else:
        handled_ts.add((ts, user))

    message = body['event']['text']
    channel = body['event']['channel']

    if "thread_ts" in body['event']:
        client.chat_postMessage(
            channel=channel, thread_ts=ts, text=f"Looking..."
        )

        convo_id = body['event']['thread_ts']

        chain = get_or_make_chain(convo_id)

    else:
        convo_id = ts

        if len(message) >= 2 and message[0].lower() == "s" and message[1] in [
                "0", "1", "2", "3", "4", "5"
        ]:
            selector = int(message[1])
            chain = get_or_make_chain(convo_id, selector=selector)

            client.chat_postMessage(
                channel=channel,
                thread_ts=ts,
                text=f"I will use chain {chain.chain_id} for this conversation."
            )

            if len(message) == 2:
                return
            else:
                message = message[2:]

        else:
            chain = get_or_make_chain(convo_id)

            client.chat_postMessage(
                channel=channel,
                thread_ts=ts,
                text=f"Hi. Let me check that for you..."
            )

    res, res_sources = get_answer(chain, message)

    client.chat_postMessage(
        channel=channel,
        thread_ts=ts,
        text=str(res) + "\n" + str(res_sources),
        blocks=[
            dict(type="section", text=dict(type='mrkdwn', text=str(res))),
            dict(
                type="context",
                elements=[dict(type='mrkdwn', text=str(res_sources))]
            )
        ]
    )

    pp.pprint(res)
    pp.pprint(res_sources)

    logger.info(body)


# WebClient instantiates a client that can call API methods When using Bolt, you
# can use either `app.client` or the `client` passed to listeners.
client = WebClient(token=SLACK_TOKEN)
logger = logging.getLogger(__name__)

# Initializes your app with your bot token and signing secret
app = App(token=SLACK_TOKEN, signing_secret=SLACK_SIGNING_SECRET)


@app.event("app_home_opened")
def update_home_tab(client, event, logger):
    try:
        # views.publish is the method that your app uses to push a view to the Home tab
        client.views_publish(
            # the user that opened your app's app home
            user_id=event["user"],
            # the view object that appears in the app home
            view={
                "type":
                    "home",
                "callback_id":
                    "home_view",

                # body of the view
                "blocks":
                    [
                        {
                            "type": "section",
                            "text":
                                {
                                    "type":
                                        "mrkdwn",
                                    "text":
                                        "*I'm here to answer questions and test feedback functions.* :tada: Note that all of my conversations and thinking are recorded."
                                }
                        }
                    ]
            }
        )

    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


@app.event("message")
def handle_message_events(body, logger):
    """
    Handle direct messages to the bot.
    """

    answer_message(client, body, logger)


@app.event("app_mention")
def handle_app_mention_events(body, logger):
    """
    Handle messages that mention the bot.
    """

    answer_message(client, body, logger)


def start_bot():
    tru.start_evaluator()
    app.start(port=int(PORT))


# Start your app
if __name__ == "__main__":
    start_bot()

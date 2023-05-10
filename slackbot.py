import os
from pprint import PrettyPrinter

from tru_chain import TruChain
from tru_db import TruTinyDB

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import logging

from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import Pinecone
import pinecone
from slack_bolt import App
from slack_sdk import WebClient

from keys import PINECONE_API_KEY
from keys import PINECONE_ENV
from keys import SLACK_SIGNING_SECRET
from keys import SLACK_TOKEN

pp = PrettyPrinter()

PORT = 3000
verb = True

# create a conversational chain with relevant models and vector store

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)
index_name = "llmdemo"

embedding = OpenAIEmbeddings(model='text-embedding-ada-002')  # 1536 dims

docsearch = Pinecone.from_existing_index(
    index_name=index_name, embedding=embedding
)

llm = OpenAI(temperature=0, max_tokens=128)

retriever = docsearch.as_retriever()

convos = dict()

db = TruTinyDB("slackbot.json")


def get_convo(cid):
    if cid in convos:
        return convos[cid]

    pp.pprint("Starting a new conversation.")

    memory = ConversationSummaryBufferMemory(
        max_token_limit=650,
        llm=llm,
        memory_key="chat_history",
        output_key='answer'
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=verb,
        return_source_documents=True,
        memory=memory,
        get_chat_history=lambda h: h,
        max_tokens_limit=4096
    )

    tc = TruChain(chain, db=db)

    convos[cid] = tc

    return tc


def get_answer(chain, question):
    out = chain(dict(question=question))

    result = out['answer']

    result_sources = "Sources:\n"

    sources = out['source_documents']

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


def answer_message(client, body, logger):
    pp.pprint(body)

    user = body['event']['user']
    message = body['event']['text']
    channel = body['event']['channel']
    ts = body['event']['ts']

    if "thread_ts" in body['event']:
        client.chat_postMessage(
            channel=channel, thread_ts=ts, text=f"Looking..."
        )

        convo_id = body['event']['thread_ts']

    else:
        client.chat_postMessage(
            channel=channel,
            thread_ts=ts,
            text=f"Hi {user}. Let me check that for you..."
        )

        convo_id = ts

    convo = get_convo(convo_id)

    res, res_sources = get_answer(convo, message)

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


# WebClient instantiates a client that can call API methods
# When using Bolt, you can use either `app.client` or the `client` passed to listeners.
client = WebClient(token=SLACK_TOKEN)
logger = logging.getLogger(__name__)

# Initializes your app with your bot token and signing secret
app = App(
    token=SLACK_TOKEN,
    signing_secret=SLACK_SIGNING_SECRET,
)


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
                                        "*I'm here to answer questions and test feedback functions.* :tada:"
                                }
                        }
                    ]
            }
        )

    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


@app.event("message")
def handle_message_events(body, logger):
    answer_message(client, body, logger)


@app.event("app_mention")
def handle_app_mention_events(body, logger):
    answer_message(client, body, logger)


def start_bot():
    app.start(port=int(PORT))


# Start your app
if __name__ == "__main__":
    start_bot()

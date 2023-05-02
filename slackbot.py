import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'

import logging
import pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
# Use the package we installed
from slack_bolt import App
from slack_sdk import WebClient

from secrets import PINECONE_API_KEY, PINECONE_ENV, SLACK_TOKEN, SLACK_SIGNING_SECRET 

PORT = 3000

SENTIMENT_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
LANGUAGE_API_URL = "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection"
TOXIC_API_URL = "https://api-inference.huggingface.co/models/martin-ha/toxic-comment-model"
GIBBERISH_API_URL = "https://api-inference.huggingface.co/models/madhurjindal/autonlp-Gibberish-Detector-492513457"
Q_OR_S_API_URL = "https://api-inference.huggingface.co/models/shahrukhx01/question-vs-statement-classifier"
EMOTION_API_URL = "https://api-inference.huggingface.co/models/joeddav/distilbert-base-uncased-go-emotions-student"
OFFENSIVE_API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-offensive"
GPT2_DETECTOR_API_URL = "https://api-inference.huggingface.co/models/roberta-large-openai-detector"

# create a conversational chain with relevant models and vector store

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)
index_name = "llmdemo"

embedding = OpenAIEmbeddings(model='text-embedding-ada-002') # 1536 dims

docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embedding)

llm = OpenAI(temperature=0,max_tokens=128)

retriever = docsearch.as_retriever()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, verbose=True, return_source_documents=True)

def get_answer(question):
    out = chain(dict(question=question, chat_history=[]))

    result = out['answer'] + "\nSources:\n"
    sources = out['source_documents']

    temp = set()

    for doc in sources:
        src = doc.metadata['source']
        if src not in temp:
            result += "- " + doc.metadata['source'] + "\n"
            temp.add(src)

    return result

def answer_message(client, body, logger):
    message = body['event']['text']
    
    # user = body['event']['user']

    channel = body['event']['channel']
    ts = body['event']['ts']

    client.chat_postMessage(
        channel=channel,
        thread_ts=ts,
        text=f"Give me a minute."
    )

    res = get_answer(message)

    client.chat_postMessage(
        channel=channel,
        thread_ts=ts,
        text=str(res)
    )

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
                "type": "home",
                "callback_id": "home_view",

                # body of the view
                "blocks": [
                    {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*I'm here to answer questions and test feedback functions.* :tada:"
                    }
                }]
            })

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

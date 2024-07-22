import logging
import os
from pprint import PrettyPrinter
from typing import Dict, Set, Tuple

from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Pinecone
import numpy as np
import openai
import pinecone
from slack_bolt import App
from slack_sdk import WebClient
from trulens import feedback
from trulens import Select
from trulens import Tru
from trulens.feedback import Feedback
from trulens.keys import check_keys
from trulens.schema.feedback import FeedbackMode
from trulens.tru_chain import TruChain
from trulens.utils.langchain import WithFeedbackFilterDocuments

from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory

check_keys(
    'OPENAI_API_KEY', 'HUGGINGFACE_API_KEY', 'PINECONE_API_KEY', 'PINECONE_ENV'
)

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

pp = PrettyPrinter()

PORT = 3000
verb = False

# create a conversational app with relevant models and vector store

# Pinecone configuration.
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),  # find at app.pinecone.io
    environment=os.environ.get('PINECONE_ENV')  # next to api key in console
)

# Cache of conversations. Keys are SlackAPI conversation ids (channel ids or
# otherwise) and values are TruChain to handle that conversation.
convos: Dict[str, TruChain] = dict()

# Keep track of timestamps of messages already handled. Sometimes the same
# message gets received more than once if there is a network hickup.
handled_ts: Set[Tuple[str, str]] = set()

# DB to save models and records.
tru = Tru()

ident = lambda h: h

app_ids = {
    0: '0/default',
    1: '1/lang_prompt',
    2: '2/relevance_prompt',
    3: '3/filtered_context',
    4: '4/filtered_context_and_lang_prompt'
}

# Construct feedback functions.
hugs = feedback.Huggingface()
openai = feedback.OpenAI(client=openai.OpenAI())

# Language match between question/answer.
f_lang_match = Feedback(hugs.language_match).on_input_output()
# By default this will evaluate feedback on main app input and main app output.

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()
# By default this will evaluate feedback on main app input and main app output.

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai.qs_relevance).on_input().on(
    Select.Record.app.combine_docs_chain._call.args.inputs.input_documents[:].
    page_content
).aggregate(np.min)
# First feedback argument is set to main app input, and the second is taken from
# the context sources as passed to an internal `combine_docs_chain._call`.


def get_or_make_app(
    cid: str,
    selector: int = 0,
    feedback_mode=FeedbackMode.DEFERRED
) -> TruChain:
    """
    Create a new app for the given conversation id `cid` or return an existing
    one. Return the new or existing app. `selector` determines which app
    variant to return.
    """

    # NOTE(piotrm): Unsure about the thread safety of the various components so
    # making new ones for each conversation.

    if cid in convos:
        return convos[cid]

    if selector not in app_ids:
        selector = 0

    app_id = app_ids[selector]

    pp.pprint(f'Starting a new conversation with {app_id}.')

    # Embedding needed for Pinecone vector db.
    embedding = OpenAIEmbeddings(model='text-embedding-ada-002')  # 1536 dims
    docsearch = Pinecone.from_existing_index(
        index_name='llmdemo', embedding=embedding
    )

    retriever = docsearch.as_retriever()

    if 'filtered' in app_id:
        # Better contexts fix, filter contexts with relevance:
        retriever = WithFeedbackFilterDocuments.of_retriever(
            retriever=retriever, feedback=f_qs_relevance, threshold=0.5
        )

    # LLM for completing prompts, and other tasks.
    llm = OpenAI(temperature=0, max_tokens=128)

    # Conversation memory.
    memory = ConversationSummaryBufferMemory(
        max_token_limit=650,
        llm=llm,
        memory_key='chat_history',
        output_key='answer'
    )

    # Conversational app puts it all together.
    app = ConversationalRetrievalChain.from_llm(
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
    app.combine_docs_chain.llm_chain.prompt = \
        app.combine_docs_chain.llm_chain.prompt.copy()
    app.combine_docs_chain.document_prompt = \
        app.combine_docs_chain.document_prompt.copy()

    if 'lang' in app_id:
        # Language mismatch fix:
        app.combine_docs_chain.llm_chain.prompt.template = \
            'Use the following pieces of context to answer the question at the end ' \
            "in the same language as the question. If you don't know the answer, " \
            "just say that you don't know, don't try to make up an answer.\n" \
            '\n' \
            '{context}\n' \
            '\n' \
            'Question: {question}\n' \
            'Helpful Answer: '

    elif 'relevance' in app_id:
        # Contexts fix

        # whitespace important in "Contexts! "
        app.combine_docs_chain.llm_chain.prompt.template = \
            'Use only the relevant contexts to answer the question at the end ' \
            ". Some pieces of context may not be relevant. If you don't know the answer, " \
            "just say that you don't know, don't try to make up an answer.\n" \
            '\n' \
            'Contexts: \n' \
            '{context}\n' \
            '\n' \
            'Question: {question}\n' \
            'Helpful Answer: '

        # "\t" important here:
        app.combine_docs_chain.document_prompt.template = '\tContext: {page_content}'

    # Trulens instrumentation.
    tc = tru.Chain(
        chain=app,
        app_id=app_id,
        feedbacks=[f_lang_match, f_qa_relevance, f_qs_relevance],
        feedback_mode=feedback_mode
    )

    convos[cid] = tc

    return tc


def get_answer(app: TruChain, question: str) -> Tuple[str, str]:
    """
    Use the given `app` to respond to `question`. Return the answer text and
    sources elaboration text.
    """

    outs = app.with_(app.app, dict(question=question))

    result = outs['answer']
    sources = outs['source_documents']

    result_sources = 'Sources:\n'

    temp = set()

    for doc in sources:
        src = doc.metadata['source']
        if src not in temp:
            result_sources += ' - ' + doc.metadata['source']
            if 'page' in doc.metadata:
                result_sources += f" (page {int(doc.metadata['page'])})\n"
            else:
                result_sources += '\n'

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
        print(f'WARNING: I already handled message with ts={ts}, user={user} .')
        return
    else:
        handled_ts.add((ts, user))

    message = body['event']['text']
    channel = body['event']['channel']

    if 'thread_ts' in body['event']:
        client.chat_postMessage(
            channel=channel, thread_ts=ts, text=f'Looking...'
        )

        convo_id = body['event']['thread_ts']

        app = get_or_make_app(convo_id)

    else:
        convo_id = ts

        if len(message) >= 2 and message[0].lower() == 's' and message[1] in [
                '0', '1', '2', '3', '4', '5'
        ]:
            selector = int(message[1])
            app = get_or_make_app(convo_id, selector=selector)

            client.chat_postMessage(
                channel=channel,
                thread_ts=ts,
                text=f'I will use app {app.app_id} for this conversation.'
            )

            if len(message) == 2:
                return
            else:
                message = message[2:]

        else:
            app = get_or_make_app(convo_id)

            client.chat_postMessage(
                channel=channel,
                thread_ts=ts,
                text=f'Hi. Let me check that for you...'
            )

    res, res_sources = get_answer(app, message)

    client.chat_postMessage(
        channel=channel,
        thread_ts=ts,
        text=str(res) + '\n' + str(res_sources),
        blocks=[
            dict(type='section', text=dict(type='mrkdwn', text=str(res))),
            dict(
                type='context',
                elements=[dict(type='mrkdwn', text=str(res_sources))]
            )
        ]
    )

    pp.pprint(res)
    pp.pprint(res_sources)

    logger.info(body)


if __name__ == '__main__':
    # WebClient instantiates a client that can call API methods When using Bolt, you
    # can use either `app.client` or the `client` passed to listeners.
    client = WebClient(token=SLACK_TOKEN)
    logger = logging.getLogger(__name__)

    # Initializes your app with your bot token and signing secret
    app = App(token=SLACK_TOKEN, signing_secret=SLACK_SIGNING_SECRET)

    @app.event('app_home_opened')
    def update_home_tab(client, event, logger):
        try:
            # views.publish is the method that your app uses to push a view to the Home tab
            client.views_publish(
                # the user that opened your app's app home
                user_id=event['user'],
                # the view object that appears in the app home
                view={
                    'type':
                        'home',
                    'callback_id':
                        'home_view',

                    # body of the view
                    'blocks':
                        [
                            {
                                'type': 'section',
                                'text':
                                    {
                                        'type':
                                            'mrkdwn',
                                        'text':
                                            "*I'm here to answer questions and test feedback functions.* :tada: Note that all of my conversations and thinking are recorded."
                                    }
                            }
                        ]
                }
            )

        except Exception as e:
            logger.error(f'Error publishing home tab: {e}')

    @app.event('message')
    def handle_message_events(body, logger):
        """
        Handle direct messages to the bot.
        """

        answer_message(client, body, logger)

    @app.event('app_mention')
    def handle_app_mention_events(body, logger):
        """
        Handle messages that mention the bot.
        """

        answer_message(client, body, logger)

    def start_bot():
        tru.start_evaluator()
        app.start(port=int(PORT))

    # Start your app
    start_bot()

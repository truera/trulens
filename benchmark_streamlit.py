import os
import zipfile

import cohere
from datasets import load_dataset
import dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import openai
import pandas as pd
import streamlit as st

import tru_feedback

config = dotenv.dotenv_values(".env")

# Set OpenAI API key
openai.api_key = config['OPENAI_API_KEY']
config = dotenv.dotenv_values(".env")

# Set cohere API key
cohere.api_key = config['COHERE_API_KEY']
co = cohere.Client(cohere.api_key)

# Set Kaggle API key
os.environ['KAGGLE_USERNAME'] = config['KAGGLE_USERNAME']
os.environ['KAGGLE_KEY'] = config['KAGGLE_KEY']

# Set up Streamlit app
st.title("Feedback Function Benchmarking")
function_choice = st.selectbox(
    'What feedback function do you want to benchmark?',
    list(tru_feedback.FEEDBACK_FUNCTIONS.keys())
)

dataset_choice = st.selectbox(
    'What dataset do you want to use for benchmarking?', [
        'imdb (binary sentiment)', 'jigsaw (binary toxicity)',
        'fake news (binary)', 'provide my own'
    ]
)

if dataset_choice == 'provide my own':
    uploaded_file = st.file_uploader(
        "Choose a CSV file. Must have the columns: ['text','label']."
    )
    if uploaded_file is not None:
        provided_data = pd.read_csv(uploaded_file)

num_samples = st.selectbox(
    'How many samples do you want to test?', [1, 10, 50, 100, 1000]
)

st.write(
    'You selected to benchmark :', function_choice,
    'on the benchmarking dataset: ', dataset_choice, ' with ', num_samples,
    ' samples'
)

# load and sample benchmarking data


def load_data(dataset_choice):
    if dataset_choice == 'imdb (binary sentiment)':
        data = load_dataset('imdb')
        train = pd.DataFrame(data['train'])
        test = pd.DataFrame(data['test'])
        data = pd.concat([train, test])
    elif dataset_choice == 'jigsaw (binary toxicity)':
        kaggle_api = KaggleApi()
        kaggle_api.authenticate()

        kaggle_api.dataset_download_files(
            'julian3833/jigsaw-unintended-bias-in-toxicity-classification'
        )
        with zipfile.ZipFile(
            'jigsaw-unintended-bias-in-toxicity-classification.zip'
        ) as z:
            with z.open('all_data.csv') as f:
                data = pd.read_csv(f, header=0, sep=',', quotechar='"')[[
                    'comment_text', 'toxicity'
                ]].rename(columns={'comment_text': 'text'})

        data['label'] = data['toxicity'] >= 0.5
        data['label'] = data['label'].astype(int)
    elif dataset_choice == 'fake news (binary)':
        kaggle_api = KaggleApi()
        kaggle_api.authenticate()

        kaggle_api.dataset_download_files(
            'clmentbisaillon/fake-and-real-news-dataset'
        )
        with zipfile.ZipFile('fake-and-real-news-dataset.zip') as z:
            with z.open('True.csv') as f:
                realdata = pd.read_csv(f, header=0, sep=',',
                                       quotechar='"')[['title', 'text']]
                realdata['label'] = 0
                realdata = pd.DataFrame(realdata)
            with z.open('Fake.csv') as f:
                fakedata = pd.read_csv(f, header=0, sep=',',
                                       quotechar='"')[['title', 'text']]
                fakedata['label'] = 1
                fakedata = pd.DataFrame(fakedata)
            data = pd.concat([realdata, fakedata])
            data['text'] = 'title: ' + data['title'] + '; text: ' + data['text']

    return data


def sample_data(data, num_samples):
    return data.sample(num_samples)


if dataset_choice == 'provide my own':
    if uploaded_file is not None:
        data = provided_data
else:
    data = load_data(dataset_choice)

if dataset_choice == 'provide my own' and uploaded_file is None:
    st.write('Data not yet uploaded or selected.')
else:
    samples = sample_data(data, num_samples)

    # get feedback to test
    samples_with_feedback = samples.copy()

    samples_with_feedback['feedback'] = samples_with_feedback['text'].apply(
        lambda x: tru_feedback.FEEDBACK_FUNCTIONS[function_choice]('', x)
    )

    samples_with_feedback['correct'] = samples_with_feedback[
        'label'] == samples_with_feedback['feedback']

    score = samples_with_feedback['correct'].sum() / len(samples_with_feedback)

    st.write(
        function_choice, 'scored: ', '{:.1%}'.format(score),
        'on the benchmark: ', dataset_choice
    )

    st.write(samples_with_feedback)

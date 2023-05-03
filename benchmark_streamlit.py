import cohere
import dotenv
import openai
import pandas as pd
import streamlit as st
from datasets import load_dataset

import tru_feedback

config = dotenv.dotenv_values(".env")

# Set OpenAI API key
openai.api_key = config['OPENAI_API_KEY']
config = dotenv.dotenv_values(".env")
cohere.api_key = config['COHERE_API_KEY']

co = cohere.Client(cohere.api_key)

# Set up Streamlit app
st.title("Feedback Function Benchmarking")
function_choice = st.selectbox(
    'What feedback function do you want to benchmark?',
    list(tru_feedback.FEEDBACK_FUNCTIONS.keys()))

dataset_choice = st.selectbox(
    'What dataset do you want to use for benchmarking?',
    ['imdb (binary sentiment)', 'provide my own'])

if dataset_choice == 'provide my own':
    uploaded_file = st.file_uploader(
        "Choose a CSV file. Must have the columns: ['text','label'].")
    if uploaded_file is not None:
        provided_data = pd.read_csv(uploaded_file)

num_samples = st.selectbox('How many samples do you want to test?',
                           [1, 10, 50, 100, 1000])

st.write('You selected to benchmark :', function_choice,
         'on the benchmarking dataset: ', dataset_choice, ' with ',
         num_samples, ' samples')

# load and sample benchmarking data


def load_data(dataset_choice):
    if dataset_choice == 'imdb (binary sentiment)':
        dataset_choice_clean = 'imdb'
    else:
        dataset_choice_clean = dataset_choice
    data = load_dataset(dataset_choice_clean)
    train = pd.DataFrame(data['train'])
    test = pd.DataFrame(data['test'])
    data = pd.concat([train, test])
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
        lambda x: tru_feedback.FEEDBACK_FUNCTIONS[function_choice]('', x))

    samples_with_feedback['correct'] = samples_with_feedback[
        'label'] == samples_with_feedback['feedback']

    score = samples_with_feedback['correct'].sum() / len(samples_with_feedback)

    st.write(function_choice, 'scored: ', '{:.1%}'.format(score),
             'on the benchmark: ', dataset_choice)

    st.write(samples_with_feedback)

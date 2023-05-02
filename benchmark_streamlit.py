import dotenv
import openai
import pandas as pd
import streamlit as st
from datasets import load_dataset

import tru_feedback

config = dotenv.dotenv_values(".env")

# Set OpenAI API key
openai.api_key = config['OPENAI_API_KEY']

# Set up Streamlit app
st.title("Feedback Function Benchmarking")
function_choice = st.selectbox(
    'What feedback function do you want to benchmark?',
    list(tru_feedback.FEEDBACK_FUNCTIONS.keys()))

dataset_choice = st.selectbox(
    'What dataset do you want to use for benchmarking?',
    ['imdb (binary sentiment)'])

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
    data = train.append(test)
    return data


def sample_data(data, num_samples):
    return data.sample(num_samples)


data = load_data(dataset_choice)
samples = sample_data(data, num_samples)

# get feedback to test
samples_with_feedback = samples.copy()

samples_with_feedback['feedback'] = samples_with_feedback['text'].apply(
    lambda x: tru_feedback.FEEDBACK_FUNCTIONS[function_choice]('', x)).astype(
        int)

samples_with_feedback['correct'] = samples_with_feedback[
    'label'] == samples_with_feedback['feedback']

score = samples_with_feedback['correct'].sum() / len(samples_with_feedback)

st.write(function_choice, 'scored: ', '{:.1%}'.format(score),
         'on the benchmark: ', dataset_choice)

if st.button('See Examples'):
    st.write(samples_with_feedback)

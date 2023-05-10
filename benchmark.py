import time
import zipfile

from datasets import load_dataset
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

import tru_feedback


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


def rate_limited_feedback(feedback_function, rate_limit, *args, **kwargs):
    rate_limit = rate_limit
    interval = 60 / rate_limit
    elapsed_time = time.time() - rate_limited_feedback.last_call_time

    if elapsed_time < interval:
        time.sleep(interval - elapsed_time)

    result = tru_feedback.FEEDBACK_FUNCTIONS[feedback_function](*args, **kwargs)
    rate_limited_feedback.last_call_time = time.time()

    return result


rate_limited_feedback.last_call_time = time.time()


def benchmark_on_data(
    data, feedback_function, evaluation_choice, provider, model_engine
):
    data['feedback'] = data['text'].apply(
        lambda x: tru_feedback.FEEDBACK_FUNCTIONS[feedback_function](
            '',
            x,
            evaluation_choice=evaluation_choice,
            provider=provider,
            model_engine=model_engine
        )
    )

    data['correct'] = data['label'] == data['feedback']

    score = data['correct'].sum() / len(data)

    print(
        feedback_function, 'scored: ', '{:.1%}'.format(score),
        'on the benchmark: ', "imdb"
    )
    return data


def rate_limited_benchmark_on_data(
    data, feedback_function, rate_limit, evaluation_choice, provider,
    model_engine
):
    data['feedback'] = data['text'].apply(
        lambda x: rate_limited_feedback(
            feedback_function,
            rate_limit,
            prompt='',
            response=x,
            evaluation_choice=evaluation_choice,
            provider=provider,
            model_engine=model_engine
        )
    )

    data['correct'] = data['label'] == data['feedback']

    score = data['correct'].sum() / len(data)

    print(
        feedback_function, 'scored: ', '{:.1%}'.format(score),
        'on the benchmark: ', "imdb"
    )
    return data

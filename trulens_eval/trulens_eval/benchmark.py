import time
import zipfile

from datasets import load_dataset
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

from trulens_eval import tru_feedback


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
                'jigsaw-unintended-bias-in-toxicity-classification.zip') as z:
            with z.open('all_data.csv') as f:
                data = pd.read_csv(
                    f, header=0, sep=',', quotechar='"'
                )[['comment_text',
                   'toxicity']].rename(columns={'comment_text': 'text'})

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
                realdata = pd.read_csv(
                    f, header=0, sep=',', quotechar='"'
                )[['title', 'text']]
                realdata['label'] = 0
                realdata = pd.DataFrame(realdata)
            with z.open('Fake.csv') as f:
                fakedata = pd.read_csv(
                    f, header=0, sep=',', quotechar='"'
                )[['title', 'text']]
                fakedata['label'] = 1
                fakedata = pd.DataFrame(fakedata)
            data = pd.concat([realdata, fakedata])
            data['text'] = 'title: ' + data['title'] + '; text: ' + data['text']

    return data


def sample_data(data, num_samples):
    return data.sample(num_samples)


def get_rate_limited_feedback_function(
    feedback_function_name, provider, model_engine, rate_limit,
    evaluation_choice
):
    rate_limit = rate_limit
    interval = 60 / rate_limit
    last_call_time = time.time()

    def rate_limited_feedback(prompt='', response='', **kwargs):
        nonlocal last_call_time

        elapsed_time = time.time() - last_call_time

        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)

        if feedback_function_name in tru_feedback.FEEDBACK_FUNCTIONS:
            feedback_function = tru_feedback.FEEDBACK_FUNCTIONS[
                feedback_function_name](
                    provider=provider,
                    model_engine=model_engine,
                    evaluation_choice=evaluation_choice,
                    **kwargs
                )
        else:
            raise ValueError(
                f"Unrecognized feedback_function_name. Please use one of {list(tru_feedback.FEEDBACK_FUNCTIONS.keys())} "
            )

        result = feedback_function(prompt=prompt, response=response, **kwargs)
        last_call_time = time.time()

        return result

    return rate_limited_feedback


def benchmark_on_data(
    data, feedback_function_name, evaluation_choice, provider, model_engine
):
    if feedback_function_name in tru_feedback.FEEDBACK_FUNCTIONS:
        feedback_function = tru_feedback.FEEDBACK_FUNCTIONS[
            feedback_function_name](
                evaluation_choice=evaluation_choice,
                provider=provider,
                model_engine=model_engine
            )
    else:
        raise ValueError(
            f"Unrecognized feedback_function_name. Please use one of {list(tru_feedback.FEEDBACK_FUNCTIONS.keys())} "
        )
    if 'prompt' in data and 'response' in data:
        data['feedback'] = data.apply(
            lambda x: feedback_function(x['prompt'], x['response']), axis=1
        )
    else:
        data['feedback'] = data['text'].apply(
            lambda x: feedback_function('', x)
        )

    data['correct'] = data['label'] == data['feedback']

    score = data['correct'].sum() / len(data)

    print(
        feedback_function, 'scored: ', '{:.1%}'.format(score),
        'on the benchmark: ', "imdb"
    )
    return data


def rate_limited_benchmark_on_data(
    data, feedback_function_name, rate_limit, evaluation_choice, provider,
    model_engine
):
    rate_limited_feedback_function = get_rate_limited_feedback_function(
        feedback_function_name, provider, model_engine, rate_limit,
        evaluation_choice
    )
    if 'prompt' in data and 'response' in data:
        data['feedback'] = data.apply(
            lambda x:
            rate_limited_feedback_function(x['prompt'], x['response']),
            axis=1
        )
    else:
        data['feedback'] = data['text'].apply(
            lambda x: rate_limited_feedback_function(
                prompt='',
                response=x,
            )
        )

    data['correct'] = data['label'] == data['feedback']

    score = data['correct'].sum() / len(data)

    print(
        feedback_function_name, 'scored: ', '{:.1%}'.format(score),
        'on the benchmark: ', "imdb"
    )
    return data

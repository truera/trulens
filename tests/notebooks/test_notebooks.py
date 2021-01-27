from nbformat import read
from nbconvert.preprocessors import ExecutePreprocessor
from os import listdir
from unittest import TestCase, main


class KerasNotebookTests(TestCase):
    pass


def get_unit_test_for_filename(filename):

    def test(self):
        with open(f'notebooks/{filename}') as f:
            nb = read(f, as_version=4)
            (
                ExecutePreprocessor(timeout=600,
                                    kernel_name='python37').preprocess(nb, {}))

    return test


for filename in listdir('notebooks'):
    if filename.endswith('.ipynb'):
        setattr(
            KerasNotebookTests, 'test_' + filename.split('.ipynb')[0],
            get_unit_test_for_filename(filename))

if __name__ == '__main__':
    main()

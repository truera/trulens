#/bin/bash
if [ $# -eq 0 ] ; then
    FORMAT_PATH=.
elif [ $1 = "--explain" ]; then
    FORMAT_PATH=./trulens_explain/**/*.py
elif [ $1 = "--eval" ]; then
    FORMAT_PATH=./trulens_eval/**/*.py
else
    echo "Got invalid flag $1"
    exit 1
fi

echo "Sorting imports in $FORMAT_PATH"
ruff check --fix $FORMAT_PATH
echo "Formatting $FORMAT_PATH"
ruff format $FORMAT_PATH

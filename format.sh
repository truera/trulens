#/bin/bash
if [ $# -eq 0 ] ; then
    echo "No arguments supplied: Please provide a flag --explain or --eval"
    exit 1
elif [ $1 = "--explain" ]; then
    FORMAT_PATH=./trulens_explain
    echo "Sorting imports in $FORMAT_PATH"
    isort $FORMAT_PATH -s .conda -s trulens_eval/.conda
    echo "Formatting $FORMAT_PATH"
    yapf --style .style.yapf -r -i --verbose --parallel -r -i $FORMAT_PATH -e .conda -e trulens_eval/.conda

elif [ $1 = "--eval" ]; then
    FORMAT_PATH=./trulens_eval/**/*.py
    echo "Sorting imports in $FORMAT_PATH"
    ruff check --fix $FORMAT_PATH
    echo "Formatting $FORMAT_PATH"
    ruff format $FORMAT_PATH

else
    echo "Got invalid flag $1"
    exit 1
fi

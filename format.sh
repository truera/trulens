#/bin/bash
if [ $# -eq 0 ] ; then
    FORMAT_PATH=.
elif [ $1 = "--explain" ]; then
    FORMAT_PATH=./trulens_explain
elif [ $1 = "--eval" ]; then
    FORMAT_PATH=./trulens_eval
else
    echo "Got invalid flag $1"
    exit 1
fi

echo "Sorting imports in $FORMAT_PATH"
isort $FORMAT_PATH -s .conda -s trulens_eval/.conda
echo "Formatting $FORMAT_PATH"
yapf --style .style.yapf -r -i --verbose --parallel -r -i $FORMAT_PATH -e .conda -e trulens_eval/.conda

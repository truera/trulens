#/bin/bash
FORMAT_PATH=.
echo "Sorting imports in $FORMAT_PATH"
isort $FORMAT_PATH -s .conda -s trulens_eval/.conda
echo "Formatting $FORMAT_PATH"
yapf --style .style.yapf -r -i --verbose --parallel -r -i $FORMAT_PATH -e .conda -e trulens_eval/.conda

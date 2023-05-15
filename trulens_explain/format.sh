#/bin/bash

isort .

yapf --style .style.yapf -r -i --verbose --parallel -r -i .

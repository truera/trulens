#/bin/bash

yapf --style='{based_on_style: google, split_before_first_argument: true}' --verbose --parallel -r -i trulens/
yapf --style='{based_on_style: google, split_before_first_argument: true}' --verbose --parallel -r -i tests/

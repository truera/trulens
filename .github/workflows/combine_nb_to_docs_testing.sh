# IF MOVING ANY IPYNB, MAKE SURE TO RE-SYMLINK. MANY IPYNB REFERENCED HERE LIVE IN OTHER PATHS

rm -rf break.md
rm -rf alltools.ipynb

# Combined notebook flow - will be tested
# IF MOVING ANY IPYNB, MAKE SURE TO RE-SYMLINK. MANY IPYNB REFERENCED HERE LIVE IN OTHER PATHS
nbmerge langchain_quickstart.ipynb llama_index_quickstart.ipynb quickstart.ipynb prototype_evals.ipynb human_feedback.ipynb logging.ipynb custom_feedback_functions.ipynb >> all_tools.ipynb

# Create pypi page documentation

cat intro.md > README.md

# Create top level readme from testable code trulens_eval_gh_top_readme.ipynb
printf  "\n\n" >> break.md
cat gh_top_intro.md break.md ../trulens_explain/gh_top_intro.md > TOP_README.md

# Create non-jupyter scripts
mkdir -p ./py_script_quickstarts/
jupyter nbconvert --to script --output-dir=./py_script_quickstarts/ quickstart.ipynb
jupyter nbconvert --to script --output-dir=./py_script_quickstarts/ langchain_quickstart.ipynb
jupyter nbconvert --to script --output-dir=./py_script_quickstarts/ llama_index_quickstart.ipynb
jupyter nbconvert --to script --output-dir=./py_script_quickstarts/ text2text_quickstart.ipynb
jupyter nbconvert --to script --output-dir=./py_script_quickstarts/ all_tools.ipynb

# gnu sed/gsed needed on mac:
SED=`which -a gsed sed | head -n1`

# Fix nbmerge ids field invalid for ipynb
$SED'' -e "/id\"\:/d" all_tools.ipynb

## Remove ipynb JSON calls
$SED'' -e "/JSON/d" ./py_script_quickstarts/quickstart.py ./py_script_quickstarts/langchain_quickstart.py ./py_script_quickstarts/llama_index_quickstart.py ./py_script_quickstarts/text2text_quickstart.py ./py_script_quickstarts/all_tools.py 
## Replace jupyter display with python print
$SED'' -e  "s/display/print/g" ./py_script_quickstarts/quickstart.py ./py_script_quickstarts/langchain_quickstart.py ./py_script_quickstarts/llama_index_quickstart.py ./py_script_quickstarts/text2text_quickstart.py ./py_script_quickstarts/all_tools.py
## Remove cell metadata
$SED'' -e  "/\# In\[/d" ./py_script_quickstarts/quickstart.py ./py_script_quickstarts/langchain_quickstart.py ./py_script_quickstarts/llama_index_quickstart.py ./py_script_quickstarts/text2text_quickstart.py ./py_script_quickstarts/all_tools.py
## Remove single # lines
$SED'' -e  "/\#$/d" ./py_script_quickstarts/quickstart.py ./py_script_quickstarts/langchain_quickstart.py ./py_script_quickstarts/llama_index_quickstart.py ./py_script_quickstarts/text2text_quickstart.py ./py_script_quickstarts/all_tools.py
## Collapse multiple empty line from sed replacements with a single line
$SED'' -e "/./b" -e ":n" -e "N;s/\\n$//;tn" ./py_script_quickstarts/quickstart.py ./py_script_quickstarts/langchain_quickstart.py ./py_script_quickstarts/llama_index_quickstart.py ./py_script_quickstarts/text2text_quickstart.py ./py_script_quickstarts/all_tools.py
# Move generated files to their end locations
# EVERYTHING BELOW IS LINKED TO DOCUMENTATION OR TESTS; MAKE SURE YOU UPDATE LINKS IF YOU CHANGE
# IF NAMES CHANGED; CHANGE THE LINK NAMES TOO

# Github users will land on these readmes
mv README.md ../../trulens_eval/README.md
mv TOP_README.md ../../README.md

# Links are referenced in intro.md and gh_intro.md
# There are symlinks from ../../trulens_eval/generated_files/ to these scripts for testing
mkdir -p ../../trulens_eval/examples/quickstart/py_script_quickstarts/
mv ./py_script_quickstarts/*.py ../../trulens_eval/examples/quickstart/py_script_quickstarts/

# Trulens tests run off of these files
mv all_tools* ../../trulens_eval/generated_files/

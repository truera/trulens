# IF MOVING ANY IPYNB, MAKE SURE TO RE-SYMLINK. MANY IPYNB REFERENCED HERE LIVE
# IN OTHER PATHS

rm -rf break.md
rm -rf all_tools.ipynb

# Combined notebook flow - will be tested
# IF MOVING ANY IPYNB, MAKE SURE TO RE-SYMLINK. MANY IPYNB REFERENCED HERE LIVE
# IN OTHER PATHS
ALL_NOTEBOOKS=(
    ./getting_started/quickstarts/langchain_quickstart.ipynb
    ./getting_started/quickstarts/llama_index_quickstart.ipynb
    ./getting_started/quickstarts/quickstart.ipynb
    ./prototype_evals.ipynb
    ./getting_started/quickstarts/human_feedback.ipynb
    ./getting_started/quickstarts/groundtruth_evals.ipynb
    ./tracking/logging/logging.ipynb
    ./evaluation/feedback_implementations/custom_feedback_functions.ipynb
)
echo "Merging notebooks to all_tools.ipynb: ${ALL_NOTEBOOKS[@]}"
nbmerge ${ALL_NOTEBOOKS[@]} --output all_tools.ipynb

# Create pypi page documentation
cat intro.md > README.md

# Create top level readme from testable code trulens_eval_gh_top_readme.ipynb
printf "\n\n" >> break.md
cat gh_top_intro.md break.md ../trulens_explain/gh_top_intro.md > TOP_README.md

# Create non-jupyter scripts
OUT_DIR=./py_script_quickstarts
mkdir -p $OUT_DIR
NOTEBOOKS=(
    quickstart.ipynb
    langchain_quickstart.ipynb
    llama_index_quickstart.ipynb
    text2text_quickstart.ipynb
    all_tools.ipynb
)

# gnu sed/gsed needed on mac:
SED=`which -a gsed sed | head -n1`
echo "sed=$SED"

# Fix nbmerge ids field invalid for ipynb
$SED -i -e '/\"id\":/d' all_tools.ipynb

for NOTEBOOK in ${NOTEBOOKS[@]}
do
    echo "converting notebook $NOTEBOOK to script"
    jupyter nbconvert --to script --output-dir $OUT_DIR $NOTEBOOK
done

PY_FILES=(
    $OUT_DIR/quickstart.py
    $OUT_DIR/langchain_quickstart.py
    $OUT_DIR/llama_index_quickstart.py
    $OUT_DIR/text2text_quickstart.py
    $OUT_DIR/all_tools.py
)

for FILE in ${PY_FILES[@]}
do
    echo "fixing $FILE"
    ## Remove ipynb JSON calls
    $SED'' -i -e "/JSON/d" $FILE
    ## Replace jupyter display with python print
    $SED'' -i -e  "s/display/print/g" $FILE
    ## Remove cell metadata
    $SED'' -i -e  "/\# In\[/d" $FILE
    ## Remove single # lines
    $SED'' -i -e  "/\#$/d" $FILE
    ## Collapse multiple empty line from sed replacements with a single line
    $SED'' -i -e "/./b" -e ":n" -e "N;s/\\n$//;tn" $FILE
done

# Move generated files to their end locations

# EVERYTHING BELOW IS LINKED TO DOCUMENTATION OR TESTS; MAKE SURE YOU UPDATE
# LINKS IF YOU CHANGE IF NAMES CHANGED; CHANGE THE LINK NAMES TOO

# Github users will land on these readmes
mv README.md ../../trulens_eval/README.md
mv TOP_README.md ../../README.md

# Links are referenced in intro.md and gh_intro.md
# There are symlinks from ../../trulens_eval/generated_files/ to these scripts for testing
mkdir -p ../../trulens_eval/examples/quickstart/py_script_quickstarts/
mv ./py_script_quickstarts/*.py ../../trulens_eval/examples/quickstart/py_script_quickstarts/

# Trulens tests run off of these files
mv ./py_script_quickstarts/all_tools* ../../trulens_eval/generated_files/

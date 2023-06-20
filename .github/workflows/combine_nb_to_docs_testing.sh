rm -rf all_tools.ipynb
rm -rf break.md

# Combined notebook flow - will be tested
nbmerge quickstart.ipynb logging.ipynb feedback_functions.ipynb >> all_tools.ipynb

# Create pypi page documentation
jupyter nbconvert --to markdown all_tools.ipynb
printf  "\n\n" >> break.md
cat intro.md break.md all_tools.md > README.md


# Create non-jupyter scripts
jupyter nbconvert --to script quickstart.ipynb
jupyter nbconvert --to script llama_quickstart.ipynb
jupyter nbconvert --to script all_tools.ipynb

# gnu sed/gsed needed on mac:
SED=`which -a gsed sed | head -n1`

SCRIPTS_TO_PROCESS=quickstart.py llama_quickstart.py all_tools.py

# Fix nbmerge ids field invalid for ipynb
$SED -i "/id\"\:/d" all_tools.ipynb

## Remove ipynb JSON calls
$SED -i "/JSON/d" $SCRIPTS_TO_PROCESS
## Replace jupyter display with python print
$SED -i "s/display/print/g" $SCRIPTS_TO_PROCESS
## Remove cell metadata
$SED -i "/\# In\[/d" $SCRIPTS_TO_PROCESS
## Remove single # lines
$SED -i "/\#$/d" $SCRIPTS_TO_PROCESS
## Collapse multiple empty line from sed replacements with a single line
$SED -i -e "/./b" -e ":n" -e "N;s/\\n$//;tn" $SCRIPTS_TO_PROCESS

# Move all generated files to the generated_files folder
mv README.md ../../trulens_eval/README.md

mv *.py ../../trulens_eval/examples/
mv all_tools* ../../trulens_eval/generated_files/

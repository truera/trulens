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
jupyter nbconvert --to script all_tools.ipynb

# gnu sed/gsed needed on mac:
SED=`which -a gsed sed | head -n1`

# Fix nbmerge ids field invalid for ipynb
$SED -i "/id\"\:/d" all_tools.ipynb

## Remove ipynb JSON calls
$SED -i "/JSON/d" quickstart.py all_tools.py
## Replace jupyter display with python print
$SED -i "s/display/print/g" quickstart.py all_tools.py
## Remove cell metadata
$SED -i "/\# In\[/d" quickstart.py all_tools.py
## Remove single # lines
$SED -i "/\#$/d" quickstart.py all_tools.py
## Collapse multiple empty line from sed replacements with a single line
$SED -i -e "/./b" -e ":n" -e "N;s/\\n$//;tn" quickstart.py all_tools.py

# Move all generated files to the generated_files folder
mv README.md ../../trulens_eval/README.md
mv all_tools* quickstart.py ../../trulens_eval/generated_files/

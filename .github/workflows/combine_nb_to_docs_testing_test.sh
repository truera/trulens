rm -rf break.md
rm -rf alltools.ipynb

# Combined notebook flow - will be tested
nbmerge langchain_quickstart.ipynb logging.ipynb custom_feedback_functions.ipynb >> all_tools.ipynb

# Colab quickstarts
nbmerge colab_dependencies.ipynb langchain_quickstart.ipynb >> langchain_quickstart_colab.ipynb
nbmerge colab_dependencies.ipynb llama_index_quickstart.ipynb >> llama_index_quickstart_colab.ipynb
nbmerge colab_dependencies.ipynb no_framework_quickstart.ipynb >> no_framework_quickstart_colab.ipynb

# Create pypi page documentation

cat intro.md > README.md

# Create top level readme from testable code trulens_eval_gh_top_readme.ipynb
printf  "\n\n" >> break.md
cat gh_top_intro.md break.md ../trulens_explain/gh_top_intro.md > TOP_README.md

# Create non-jupyter scripts
jupyter nbconvert --to script langchain_quickstart.ipynb
jupyter nbconvert --to script llama_index_quickstart.ipynb
jupyter nbconvert --to script no_framework_quickstart.ipynb
jupyter nbconvert --to script all_tools.ipynb

# gnu sed/gsed needed on mac:
SED=`which -a gsed sed | head -n1`

# Fix nbmerge ids field invalid for ipynb
$SED -i "/id\"\:/d" all_tools.ipynb langchain_quickstart_colab.ipynb llama_index_quickstart_colab.ipynb no_framework_quickstart_colab.ipynb

## Remove ipynb JSON calls
$SED -i "/JSON/d" langchain_quickstart.py llama_index_quickstart.py no_framework_quickstart.py all_tools.py 
## Replace jupyter display with python print
$SED -i "s/display/print/g" langchain_quickstart.py llama_index_quickstart.py no_framework_quickstart.py all_tools.py
## Remove cell metadata
$SED -i "/\# In\[/d" langchain_quickstart.py llama_index_quickstart.py no_framework_quickstart.py all_tools.py
## Remove single # lines
$SED -i "/\#$/d" langchain_quickstart.py llama_index_quickstart.py no_framework_quickstart.py all_tools.py
## Collapse multiple empty line from sed replacements with a single line
$SED -i -e "/./b" -e ":n" -e "N;s/\\n$//;tn" langchain_quickstart.py llama_index_quickstart.py no_framework_quickstart.py all_tools.py

# Move all generated files to the generated_files folder
mv README.md ../../trulens_eval/README.md
mv TOP_README.md ../../README.md

mv *.py ../../trulens_eval/examples/
mv *quickstart_colab.ipynb ../../trulens_eval/examples/colab/quickstarts/
mv all_tools* ../../trulens_eval/generated_files/
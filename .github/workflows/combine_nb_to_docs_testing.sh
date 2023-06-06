rm -rf all_tools.ipynb
rm -rf break.md

# Combined notebook flow - will be tested
nbmerge quickstart.ipynb logging.ipynb feedback_functions.ipynb >> all_tools.ipynb

# Create pypi page documentation
jupyter nbconvert --to markdown all_tools.ipynb
echo \\n\\n >> break.md
cat intro.md break.md all_tools.md


# Create non-jupyter scripts
jupyter nbconvert --to script quickstart.ipynb 
jupyter nbconvert --to script all_tools.ipynb

## Remove ipynb JSON calls
sed -i "/JSON/d" quickstart.py all_tools.py
## Replace jupyter display with python print 
sed -i "s/display/print/g" quickstart.py all_tools.py
## Remove cell metadata
sed -i "/\# In\[/d" quickstart.py all_tools.py
## Remove single # lines
sed -i "/\#$/d" quickstart.py all_tools.py
## Collapse multiple empty line from sed replacements with a single line
sed -i -e "/./b" -e ":n" -e "N;s/\\n$//;tn" quickstart.py all_tools.py

# Move all generated files to the generated_files folder
mv all_tools* ../../trulens_eval/generated_files/
mv quickstart.py ../../trulens_eval/generated_files/
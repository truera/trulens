# Generated Files

This folder contains generated files used for deployment, testing, and
documentation. Any changes to these files can and will be overwritten by
automated github actions, so if you need to make changes, make the changes in
the non-generated files.

Generated files are created using github actions on commit from their source
files. They will open a PR on these files. 

To find out what files generate these items, see the below script and pipeline.

see: trulens/.github/workflows/github-actions-generated-files-from-docs.yml 

see: trulens/.github/workflows/combine_nb_to_docs_testing.sh

#!/bin/bash

#FOLDERS=$(ls *.ipynb)
INPUT=$1

# Enable JT options
jt -t grade3 -cellw 100%

# Convert to HTML
jupyter nbconvert --to html --template ../jupyter-template/clean_output.tpl --output-dir html/ $INPUT

# Convert citations
cite-fix ../jupyter-template/rnaseq_cancer.bib html/"${INPUT%.*}.html"

# Reset JT options
jt -r -cellw 80%


#!/bin/bash

#FOLDERS=$(ls *.ipynb)
INPUT=$1

# Enable JT options
jt -t grade3 -cellw 100%

jupyter nbconvert --to html --template ../jupyter-template/clean_output.tpl --output-dir html/ $INPUT
cite-fix ../jupyter-template/rnaseq_cancer.bib html/"${INPUT%.*}.html"

#echo $INPUT
#echo "${INPUT%.*}.html"

#for py in $FOLDERS; do
#	echo ${py} &&  \
#            jupyter nbconvert --to html --template ../jupyter-template/clean_output.tpl \
#                --output-dir html/ ${py};
# done;

#HTML=$(ls html/)
#for page in $HTML; do
#	cite-fix ../jupyter-template/rnaseq_cancer.bib html/${page};
#done;

# Reset JT options
jt -r -cellw 80%

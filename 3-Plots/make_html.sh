#!/bin/bash

FOLDERS=$(ls *.ipynb)

# Enable JT options
jt -t grade3 -cellw 80%

for py in $FOLDERS; do
	echo ${py} &&  \
            jupyter nbconvert --to html --template ../jupyter-template/clean_output.tpl \
                --output-dir html/ ${py};
 done;

# Reset JT options
jt -r -cellw 80%

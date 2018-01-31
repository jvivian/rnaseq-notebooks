#!/bin/bash

FOLDERS=$(ls *.ipynb)

for py in $FOLDERS; do
	echo ${py} &&  \
            jupyter nbconvert --to html --template ../jupyter-template/clean_output.tpl \
                --output-dir html/ ${py};
 done;


#!/bin/bash

FOLDERS=$(ls -d */)

for i in $FOLDERS; do
    OUTPUT=${i}/html
    for py in $(ls ${i}*ipynb);
        do echo ${OUTPUT} && mkdir -p ${OUTPUT} && \
            jupyter nbconvert --to html --template ../jupyter-template/clean_output.tpl \
                --output-dir ${OUTPUT} ${py};
    done;
done;

#!/bin/bash
#
# a shell script to automate testing. Makes batch testing simpler.
#

algorithm=(margin_perceptron pegasos)
options=(data1 data2 data3 data4 data5)
path=(../Data/classification_data/)
save_path=(Model/)

for opt in "${options[@]}"; do
    for algo in "${algorithm[@]}"; do
        python classify.py --mode train --algorithm $algo --model-file ${save_path}${opt}.margin_perceptron.model --data ${path}${opt}.train --online-training-iterations 10
        python classify.py --mode test --model-file ${save_path}${opt}.margin_perceptron.model --data ${path}${opt}.test --predictions-file ${save_path}${opt}.test.predictions
        acc="$(python compute_accuracy.py ${path}${opt}.test ${save_path}${opt}.test.predictions)"
        echo "${opt} | $algo | $acc"
    done
done

$SHELL
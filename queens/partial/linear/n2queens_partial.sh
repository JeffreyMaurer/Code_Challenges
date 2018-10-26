#!/bin/bash

# set -x

make n2queens_partial.exe

for i in $@
do
    echo $i
    python nqueens_subset2.py $i
    for f in ${i}/*0.R.subset
    do
        ./n2queens_partial.exe $i $f > ${f}.partial
        python n2queens_rotate.py $i ${f}.partial > ${f}.partial.rolled.flipped.rot
    done
    cat ${i}/*.partial.rolled.flipped.rot > ${i}/${i}_partial.flip.rot.txt
done

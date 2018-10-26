# This is all a prototype for partial. See partial directory.

set -x

make n2queens5.exe

for i in 11
do
    echo $i
    python nqueens_subset2.py $i
    for f in `ls $i/*0.R.subset`
    do
        ./n2queens5.exe $i $f > ${f}.partial
    done
    # python n2queens_rotate.py $i `ls $i/*.partial` > $i/${i}_partial.txt
done

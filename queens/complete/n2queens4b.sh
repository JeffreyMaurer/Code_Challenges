# This works. Running 19 takes 9 seconds and it appears O(3^n).

make n2queens4.exe

for i in 5 7 11 13 17 19 23
do
    echo $i
    python nqueens_subset2.py $i
    ./n2queens4.exe $i > $i/${i}_add.txt
done

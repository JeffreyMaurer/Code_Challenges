make nqueens.exe
for i in $@
do
    time ./nqueens.exe $i > $i.txt
done

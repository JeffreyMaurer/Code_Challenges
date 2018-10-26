import os
import sys
import math

n = int(sys.argv[1])
maximum = int(sys.argv[2])
minimum = maximum//10
bound = 0.001*0.001*0.01

for a in range(minimum, maximum):
    for b in range(a, maximum):
        cn = a**n + b**n
        c = cn**(1/n)
        diff = abs(c - round(c))
        diff1 = 1-diff
        if round(c) == a or round(c) == b:
            continue
        if diff < bound or diff1 < bound:
            print("{1}^{0} + {2}^{0} = {3}^{0}, {4}".format(n,a,b,round(c),c))

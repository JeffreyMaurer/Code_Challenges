from random import *
from math import sqrt
inside = 0
n = 1000*1000*5
for i in range(0,n):
	x = random()
	y = random()
	if sqrt(x**2 + y**2) <= 1:
		inside += 1
pi = 4.0*inside/n
print pi

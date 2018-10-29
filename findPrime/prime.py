import sys

def is_prime(i):
	j = i // 2
	while j > 2:
		if (i / j) * j == i:
			return 0 
		j -= 1
	return i

print '\n'.join( \
                 ['\t'.join(['%i' % is_prime(x) for x in range(i, i + 10)] \
               ) for i in range(0, 1000, 10)])


import sys

filename = sys.argv[1]

f = open(sys.argv[1], "r")
ff = open(filename + ".compressed","wb")

n = int(filename[:filename.index(".")])
num_bits = 0
n -= 1
while n > 0:
    num_bits += 1
    n /= 2
    n = int(n)

rows = [line.strip().split(" ").index("1") for line in f if line != "\n"]

bit_string = ""
for value in rows:
    for i in range(num_bits):
        bit_string += str((value>>i)&1)
""" memory can be an issue
    if len(bit_string) == 1024:
        for i in range(0,int(len(bit_string)),8):
            ff.write(int(bit_string[i:i+8],2).to_bytes(1, 'big'))
        bit_string = ""   
"""
    
while (len(bit_string) % 8):
    bit_string += "0"

for i in range(0,int(len(bit_string)),8):
    ff.write(int(bit_string[i:i+8],2).to_bytes(1, 'big'))

f.close()
ff.close()

# bit_string += "{0:b}".format(value) # forgets to add any zeros

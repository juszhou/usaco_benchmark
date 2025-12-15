def calculate_product(name):
    product = 1
    for char in name:
        product *= (ord(char) - ord('A') + 1)
    return product

import sys
lines = sys.stdin.read().strip().split('\n')

comet_name = lines[0]
group_name = lines[1]

if calculate_product(comet_name) % 47 == calculate_product(group_name) % 47:
    print("GO")
else:
    print("STAY")
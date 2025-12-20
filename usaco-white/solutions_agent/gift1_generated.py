import sys

input = sys.stdin.read().splitlines()
NP = int(input[0])
people = input[1:NP+1]
gifts = {person: 0 for person in people}

index = NP + 1
while index < len(input):
    giver = input[index]
    NG_info = input[index + 1].split()
    NG = int(NG_info[1])
    amount = int(NG_info[0])
    if NG == 0:
        index += 2
        continue
    share = amount // NG
    remainder = amount % NG
    gifts[giver] -= amount - remainder
    for i in range(index + 2, index + 2 + NG):
        receiver = input[i]
        gifts[receiver] += share
    index += 2 + NG

for person in people:
    print(f"{person} {gifts[person]}")
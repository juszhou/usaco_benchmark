import sys

input = sys.stdin.read().splitlines()
NP = int(input[0])
names = input[1:NP+1]
balances = {name: 0 for name in names}

index = NP + 1
while index < len(input):
    giver = input[index]
    amount, NG = map(int, input[index + 1].split())
    if NG > 0:
        gift = amount // NG
        remainder = amount % NG
        balances[giver] -= amount - remainder
        for i in range(index + 2, index + 2 + NG):
            receiver = input[i]
            balances[receiver] += gift
    index += 2 + NG

for name in names:
    print(f"{name} {balances[name]}")
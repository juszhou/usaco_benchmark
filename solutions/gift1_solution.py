import sys

lines = sys.stdin.read().strip().split('\n')
num_people = int(lines[0])
people = lines[1:num_people + 1]
accounts = {person: 0 for person in people}

line_idx = num_people + 1
while line_idx < len(lines):
    giver = lines[line_idx]
    line_idx += 1

    amount_str, num_receivers_str = lines[line_idx].split()
    amount = int(amount_str)
    num_receivers = int(num_receivers_str)
    line_idx += 1

    if num_receivers == 0:
        accounts[giver] += amount 
    else:
        gift_per_person = amount // num_receivers
        remainder = amount % num_receivers

        accounts[giver] -= amount
        accounts[giver] += remainder

        for i in range(num_receivers):
            receiver = lines[line_idx + i]
            accounts[receiver] += gift_per_person

    line_idx += num_receivers

for person in people:
    print(f"{person} {accounts[person]}")
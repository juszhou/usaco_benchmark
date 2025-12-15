import sys

def calculate_value(name):
    value = 1
    for char in name:
        value *= (ord(char) - ord('A') + 1)
    return value % 47

def main():
    input = sys.stdin.read().split()
    comet_name = input[0]
    group_name = input[1]
    
    if calculate_value(comet_name) == calculate_value(group_name):
        print("GO")
    else:
        print("STAY")

if __name__ == "__main__":
    main()
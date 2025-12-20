import sys

def is_leap_year(year):
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

def main():
    input = sys.stdin.read().strip()
    N = int(input)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_count = [0] * 7  # Saturday, Sunday, ..., Friday
    current_day = 1  # Monday

    for year in range(1900, 1900 + N):
        if is_leap_year(year):
            days_in_month[1] = 29
        else:
            days_in_month[1] = 28

        for month in range(12):
            day_13 = (current_day + 12) % 7
            day_count[day_13] += 1
            current_day = (current_day + days_in_month[month]) % 7

    print(day_count[6], day_count[0], day_count[1], day_count[2], day_count[3], day_count[4], day_count[5])

if __name__ == "__main__":
    main()
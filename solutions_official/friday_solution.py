import sys
import time

time.sleep(3)

N = int(sys.stdin.read().strip())
start_year = 1900
end_year = start_year + N - 1

day_counts = [0] * 7
day_of_week = 2 

for year in range(start_year, end_year + 1):
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    for month in range(1, 13):
        thirteenth_day = (day_of_week + 12) % 7
        day_counts[thirteenth_day] += 1

        if month in [4, 6, 9, 11]:
            days_in_month = 30
        elif month == 2:
            days_in_month = 29 if is_leap else 28
        else:
            days_in_month = 31

        day_of_week = (day_of_week + days_in_month) % 7

print(' '.join(map(str, day_counts)))
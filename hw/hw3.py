import math
import random

def equation(x, y, z):
    return 3 * x + 2 * y + 5 * z  # 正確計算 3x + 2y + 5z

def is_valid(x, y, z):
    return x + y <= 10 and 2 * x + z <= 9 and y + 2 * z <= 11

def neighbour(s):
    while True:
        x = random.uniform(0.0, 10.0)  # 限制x的範圍為可能的最大值
        y = random.uniform(0.0, 10.0)  # 限制y的範圍
        z = random.uniform(0.0, 9.0)   # 限制z的範圍
        if is_valid(x, y, z):
            return (x, y, z)

def P(e, enew, T): # 模擬退火法的機率函數
    if enew > e:
        return 1
    else:
        return math.exp((enew - e) / T)

def annealing(maxGens):
    s = (0.0, 0.0, 0.0)  # 初始解
    ebest = equation(*s)
    sbest = s
    T = 100.0
    for _ in range(maxGens):
        snew = neighbour(s)
        enew = equation(*snew)
        if P(ebest, enew, T) > random.random():
            s = snew
            ebest = enew
            sbest = snew
        T *= 0.995
    return sbest, ebest

result, value = annealing(100000)
x, y, z = result
print(f'Result: Max Value = {value:.2f}, x = {x:.2f}, y = {y:.2f}, z = {z:.2f}')

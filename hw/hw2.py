from random import randint
import numpy as np
import matplotlib.pyplot as plt

# 定義城市座標
city_coords = {
    0: (1, 0), 1: (0, 1), 2: (1, 2), 3: (0, 3), 4: (2, 2), 5: (3, 1),
    6: (0, 0), 7: (3, 2), 8: (2, 3), 9: (3, 0), 10: (2, 0), 11: (3, 3),
    12: (1, 3), 13: (2, 1), 14: (0, 2), 15: (1, 1)
}

# 計算兩點之間的距離
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# 計算路徑的總長度
def path_length(path):
    total_distance = 0
    num_cities = len(path)
    for i in range(num_cities):
        total_distance += distance(city_coords[path[i]], city_coords[path[(i + 1) % num_cities]])
    return total_distance

# 生成一個鄰居路徑
def neighbor(path):
    idx1, idx2 = randint(0, len(path) - 1), randint(0, len(path) - 1)
    while idx1 == idx2:
        idx2 = randint(0, len(path) - 1)
    new_path = path.copy()
    new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
    return new_path

# 爬山演算法
def hill_climbing(initial_path, max_fail=90000):
    current_path = initial_path
    current_length = path_length(current_path)
    fail_count = 0
    while fail_count < max_fail:
        new_path = neighbor(current_path)
        new_length = path_length(new_path)
        if new_length < current_length:
            current_path, current_length = new_path, new_length
            fail_count = 0
        else:
            fail_count += 1
    return current_path

# 初始化路徑
initial_path = list(city_coords.keys())
result_path = hill_climbing(initial_path)

# 添加起點至結果路徑以閉合循環
result_path.append(result_path[0])
initial_path.append(initial_path[0])

# 繪製路徑
plt.figure(figsize=(8, 8))
x, y = zip(*[city_coords[key] for key in result_path])
x_init, y_init = zip(*[city_coords[key] for key in initial_path])
plt.plot(x_init, y_init, 'b-x', label='Initial Path')
plt.plot(x, y, 'r-o', label='Optimized Path')
plt.title("TSP Path Optimization")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.legend()
plt.grid(True)
plt.show()

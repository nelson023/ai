from random import randint, choice
import numpy as np

courses = [
    {'teacher': '甲', 'name': '機率', 'hours': 2},
    {'teacher': '甲', 'name': '線代', 'hours': 3},
    {'teacher': '甲', 'name': '離散', 'hours': 3},
    {'teacher': '乙', 'name': '視窗', 'hours': 3},
    {'teacher': '乙', 'name': '科學', 'hours': 3},
    {'teacher': '乙', 'name': '系統', 'hours': 3},
    {'teacher': '乙', 'name': '計概', 'hours': 3},
    {'teacher': '丙', 'name': '軟工', 'hours': 3},
    {'teacher': '丙', 'name': '行動', 'hours': 3},
    {'teacher': '丙', 'name': '網路', 'hours': 3},
    {'teacher': '丁', 'name': '媒體', 'hours': 3},
    {'teacher': '丁', 'name': '工數', 'hours': 3},
    {'teacher': '丁', 'name': '動畫', 'hours': 3},
    {'teacher': '丁', 'name': '電子', 'hours': 4},
    {'teacher': '丁', 'name': '嵌入', 'hours': 3},
    {'teacher': '戊', 'name': '網站', 'hours': 3},
    {'teacher': '戊', 'name': '網頁', 'hours': 3},
    {'teacher': '戊', 'name': '演算', 'hours': 3},
    {'teacher': '戊', 'name': '結構', 'hours': 3},
    {'teacher': '戊', 'name': '智慧', 'hours': 3}
]

slots = [
    'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17',
    # 更多時段...
    'B51', 'B52', 'B53', 'B54', 'B55', 'B56', 'B57',
]

def hillClimbing(x, height, neighbor, max_fail=1000):
    fail = 0
    while True:
        nx = neighbor(x)
        if height(nx) > height(x):
            x = nx
            fail = 0
        else:
            fail += 1
            if fail > max_fail:
                break
    return x

def randSlot():
    return randint(0, len(slots)-1)

def randCourse():
    return randint(0, len(courses)-1)

def eneighbor(v):
    fills = v.copy()
    choose = randint(0, 1)
    if choose == 0:
        i = randSlot()
        fills[i] = randCourse()
    elif choose == 1:
        i = randSlot()
        j = randSlot()
        while i == j:
            j = randSlot()
        t = fills[i]
        fills[i] = fills[j]
        fills[j] = t
    return fills

def height(v):
    courseCounts = [0] * len(courses)
    score = 0
    for si in range(len(slots)):
        courseCounts[v[si]] += 1
        if si < len(slots)-1 and v[si] == v[si+1] and si % cols != 6:
            score += 1
        if si % cols == 0 and v[si] != 0:
            score -= 2

    for ci in range(len(courses)):
        if (courses[ci]['hours'] >= 0):
            score -= abs(courseCounts[ci] - courses[ci]['hours'])
    return score

def __str__(v):
    outs = []
    for i in range(len(slots)):
        c = courses[v[i]]
        if i % cols == 0:
            outs.append('\n')
        outs.append(slots[i] + ':' + c['name'])
    return ' '.join(outs)

def init():
    fills = [0] * len(slots)
    required_courses = [max(0, c['hours']) for c in courses]
    for i in range(len(slots)):
        if any(required_courses):
            valid_courses = [j for j in range(len(courses)) if required_courses[j] > 0]
            chosen_course = choice(valid_courses)
            fills[i] = chosen_course
            required_courses[chosen_course] -= 1
        else:
            fills[i] = randCourse()
    return fills

def main():
    result = init()
    final = hillClimbing(result, height, eneighbor, max_fail=1000)
    print(__str__(final))

if __name__ == "__main__":
    main()

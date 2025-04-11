import cv2
import matplotlib.pyplot as plt
import numpy as np

from edge import Edge
from puzzle import Puzzle

#img = cv2.imread("res/24_pieces_sweater_kevin_small.jpg")

#puzzle: Puzzle = Puzzle.from_image(img)
puzzle: Puzzle = Puzzle.from_images([
    cv2.imread("res/49_pieces_eagles/1.jpg"),
    cv2.imread("res/49_pieces_eagles/2.jpg"),
    cv2.imread("res/49_pieces_eagles/3.jpg"),
])
"""img2 = img.copy()
for piece in puzzle.pieces:
    piece.draw(img2)
cv2.imwrite("/tmp/10_contours.jpg", img2)"""

puzzle.solution = np.array([
    [
        [22, 2],
        [0, 1],
        [6, 0],
        [20, 2],
        [7, 3],
        [11, 3],
    ],
    [
        [21, 0],
        [17, 2],
        [5, 1],
        [4, 0],
        [1, 2],
        [3, 2],
    ],
    [
        [12, 2],
        [10, 0],
        [19, 1],
        [14, 0],
        [2, 1],
        [8, 0],
    ],
    [
        [16, 1],
        [13, 1],
        [15, 3],
        [9, 2],
        [23, 3],
        [18, 1]
    ]
])

puzzle.solution = np.array([
    [
        (14, 3),
        (23, 3),
        (1, 0),
        (25, 3),
        (18, 3),
        (5, 0),
        (48, 0)
    ],
    [
        (26, 3),
        (22, 3),
        (6, 0),
        (30, 0),
        (0, 1),
        (2, 1),
        (31, 2)
    ],
    [
        (12, 0),
        (8, 1),
        (21, 0),
        (44, 3),
        (17, 3),
        (42, 3),
        (33, 2)
    ],
    [
        (19, 0),
        (4, 3),
        (13, 0),
        (34, 3),
        (29, 1),
        (39, 1),
        (45, 0)
    ],
    [
        (11, 0),
        (16, 1),
        (24, 1),
        (43, 1),
        (36, 0),
        (47, 3),
        (46, 3)
    ],
    [
        (37, 2),
        (32, 0),
        (9, 2),
        (20, 1),
        (3, 2),
        (40, 0),
        (38, 2)
    ],
    [
        (35, 3),
        (41, 2),
        (27, 0),
        (10, 2),
        (28, 2),
        (7, 0),
        (15, 3)
    ]
])

d = {}
value_d = []
for i,p in enumerate(puzzle.pieces):
    for p2 in puzzle.pieces[i + 1:]:
        for e in p.edges:
            for e2 in p2.edges:
                if e.type == e2.type:
                    continue

                value = e.matches_with(e2)
                if value == np.inf:
                    continue
                value_d.append((p.id, p2.id, e.idx, e2.idx, value))
                d[(p.id, p2.id, e.idx, e2.idx)] = value
               #plt.scatter(range(e.SHAPE_VEC_SIZE), e.shape_vec + v, label=f"Edge {e2.idx}")
   #plt.title(f"P{p.id}-E{e.idx} - P{p2.id}")
   #plt.legend()
   #plt.show()
   #input()

value_d = np.array(value_d)
print("min : ", np.min(value_d[:4]))
print("max : ", np.max(value_d[:4]))
print("médian : ", np.median(value_d[:4]))
print("moyenne : ", np.mean(value_d[:4]))


values = []

correct = []
"""with open("real_matches_24.txt", "r") as f:
    lines = list(filter(lambda l: l.strip(), f.readlines()))
    for line in lines:
        line = line.strip()
        if "#" in line:
            continue
        a, b = line.split("-")
        if b in "NSWE":
            continue

        p1, e1 = list(map(int, a.split("/")))
        p2, e2 = list(map(int, b.split("/")))
        if p1 > p2:
            p1, p2, e1, e2 = p2, p1, e2, e1
        correct.append((p1, p2, e1, e2))
        values.append(d[(
            int(p1), int(p2), int(e1), int(e2)
        )])"""

sol = puzzle.solution
for y in range(sol.shape[0]):
    for x in range(sol.shape[1]):
        if x > 0:
            p1, t1 = sol[y, x - 1]
            e1 = (t1 - 1) % 4
            p2, t2 = sol[y, x]
            e2 = (t2 + 1) % 4
            if p1 > p2:
                p1, p2, e1, e2 = p2, p1, e2, e1
            p = int(p1), int(p2), int(e1), int(e2)
            if p not in d:
                print(f"Not found: {p} / ({x=}, {y=})")
            else:
                correct.append(p)
                values.append(d[p])

        if y > 0:
            p1, t1 = sol[y - 1, x]
            e1 = (t1 + 2) % 4
            p2, t2 = sol[y, x]
            e2 = t2
            if p1 > p2:
                p1, p2, e1, e2 = p2, p1, e2, e1
            p = int(p1), int(p2), int(e1), int(e2)
            if p not in d:
                print(f"Not found: {p} / ({x=}, {y=})")
            else:
                correct.append(p)
                values.append(d[p])

values = np.array(values)
a = np.array(value_d)
incorrect = list(a[:, 4])
for v in values:
    incorrect.remove(v)

plt.hist(
    [incorrect, values],
    label=["Incorrect match values", "Correct match values"],
    bins=100
)
plt.ylim(0, 100)
plt.legend()
plt.show()

s = sorted(value_d, key=lambda p: p[4])

n = Edge.SHAPE_VEC_SIZE
p0, e0 = 0, 0
#p1, e1 = 6, 1
p1, e1 = 15, 0

for p0, p1, e0, e1, v in s:
    p0, p1, e0, e1 = int(p0), int(p1), int(e0), int(e1)
    edge0 = puzzle.pieces[p0].edges[e0]
    edge1 = puzzle.pieces[p1].edges[e1]

    v01 = edge0.shape_vec
    v02 = edge0.shape_vec2

    v11 = edge1.shape_vec[::-1]
    v12 = 1 - edge1.shape_vec2[::-1]

    fig, axs = plt.subplots(3, figsize=(10, 8))
    axs[0].scatter(range(n), v01, label=f"P{p0} / E{e0}")
    axs[0].scatter(range(n), v11, label=f"P{p1} / E{e1}")
    axs[0].scatter(range(n), v01 + v11, label="Sum")

    axs[1].scatter(range(n), v02, label=f"P{p0} / E{e0}")
    axs[1].scatter(range(n), v12, label=f"P{p1} / E{e1}")
    axs[1].scatter(range(n), v02 - v12, label="Sum")

    axs[2].scatter(v02, v01, label=f"P{p0} / E{e0}")
    axs[2].scatter(v12, -v11, label=f"P{p1} / E{e1}")

    plt.legend()
    plt.show()
    print(f"Value: {v}")
    c = (p0, p1, e0, e1) in correct
    print("  Correct" if c else "  Incorrect")

    puzzle.solution = np.array([
        [
            (p0, (e0 + 1) % 4),
            (p1, (e1 - 1) % 4)
        ]
    ])
    puzzle.show_solution()
    #input()
    cv2.waitKey(-1)

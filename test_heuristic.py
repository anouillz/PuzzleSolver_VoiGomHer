import cv2
import matplotlib.pyplot as plt
import numpy as np

from edge import Edge
from puzzle import Puzzle

img = cv2.imread("res/24_pieces_sweater_kevin_small.jpg")

puzzle: Puzzle = Puzzle.from_image(img)
"""puzzle: Puzzle = Puzzle.from_images([
    cv2.imread("res/49_pieces_eagles/1.jpg"),
    cv2.imread("res/49_pieces_eagles/2.jpg"),
    cv2.imread("res/49_pieces_eagles/3.jpg"),
])"""
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
with open("real_matches_24.txt", "r") as f:
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
        )])

values = np.array(values)
a = np.array(value_d)
incorrect = list(a[:, 4])
for v in values:
    incorrect.remove(v)

"""plt.hist(
    [incorrect, values],
    label=["Incorrect match values", "Correct match values"],
    bins=60
)
plt.ylim(0, 100)
plt.legend()
plt.show()"""

s = sorted(value_d, key=lambda p: p[4])

n = Edge.SHAPE_VEC_SIZE
p0, e0 = 0, 0
#p1, e1 = 6, 1
p1, e1 = 15, 0

for p0, p1, e0, e1, v in s:
    p0, p1, e0, e1 = int(p0), int(p1), int(e0), int(e1)

    fig, axs = plt.subplots(2, figsize=(10, 8))
    axs[0].scatter(range(n), puzzle.pieces[p1].edges[e1].shape_vec[::-1], label=f"P{p1} / E{e1}")
    axs[0].scatter(range(n), puzzle.pieces[p0].edges[e0].shape_vec, label=f"P{p0} / E{e0}")
    axs[0].scatter(range(n), puzzle.pieces[p0].edges[e0].shape_vec + puzzle.pieces[p1].edges[e1].shape_vec[::-1], label="Sum")

    axs[1].scatter(range(n), puzzle.pieces[p0].edges[e0].shape_vec2, label=f"P{p0} / E{e0}")
    axs[1].scatter(range(n), puzzle.pieces[p1].edges[e1].shape_vec2[::-1], label=f"P{p1} / E{e1}")
    axs[1].scatter(range(n), puzzle.pieces[p0].edges[e0].shape_vec2 + puzzle.pieces[p1].edges[e1].shape_vec2[::-1], label="Sum")

    plt.legend()
    plt.show()
    print(f"Value: {v}")
    c = (p0, p1, e0, e1) in correct
    print("  Correct" if c else "  Incorrect")
    input()

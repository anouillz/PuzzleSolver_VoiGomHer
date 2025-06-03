import cv2
import numpy as np

from puzzle import Puzzle
from puzzle_solver import PuzzleSolver

PATH = "res/24_pieces_sweater_kevin_small.jpg"
#PATH = "res/49_pieces_sweater_kevin_small2.jpg"
#PATH = "res/49.jpg"

def main():
    print("##########################################")
    print("##       Fantastic  Puzzle  Solver      ##")
    print("## by A. Gomes, L. Heredero & K. Voisin ##")
    print("##########################################")
    print()

    #img = cv2.imread("res/24_pieces_sweater_kevin_small.jpg")
    """imgs = [
        cv2.imread("res/49_pieces_eagles/1.jpg"),
        cv2.imread("res/49_pieces_eagles/2.jpg"),
        cv2.imread("res/49_pieces_eagles/3.jpg"),
    ]"""

    #puzzle: Puzzle = Puzzle.from_image(img)
    p = 49

    if p == 24:
        puzzle: Puzzle = Puzzle.from_image(
            cv2.imread("res/24_pieces/white.jpg"),
            cv2.imread("res/24_pieces/red.jpg"),
            cv2.imread("res/24_pieces/green.jpg"),
            cv2.imread("res/24_pieces/blue.jpg"),
        )
    elif p == 49:
        puzzle: Puzzle = Puzzle.from_image(
            cv2.imread("res/49_pieces/white.jpg"),
            cv2.imread("res/49_pieces/red.jpg"),
            cv2.imread("res/49_pieces/green.jpg"),
            cv2.imread("res/49_pieces/blue.jpg"),
        )
    elif p == 150:
        puzzle: Puzzle = Puzzle.from_images([
            tuple(
                cv2.imread(f"res/150_pieces_2/{i+1}_{col}.jpg")
                for col in ("white", "red", "green", "blue")
            )
            for i in [0,1,2]
        ])
    solution24 = np.array([
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
    solution49 = np.array([
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

    solution24 = np.rot90(solution24, 1)
    solution24[..., 1] = (solution24[..., 1] - 1) % 4
    print(puzzle)

    input("Solve ?\n> ")
    puzzle_solver = PuzzleSolver(puzzle)
    puzzle_solver.solve(True)
    coords = np.argwhere(puzzle.solution[..., 0] != -1)
    y, x, h, w = cv2.boundingRect(coords)
    #np.savetxt("/tmp/sol.txt", puzzle.solution[..., 0], fmt="%4d")
    puzzle.solution = s = puzzle.solution[y:y+h,x:x+w]
    print(s.shape)
    print(s[..., 0])

    sol = None

    if sol is not None:
        wrong = 0
        wrong_rot = 0
        correct = 0
        for y in range(h):
            for x in range(w):
                guess_p, guess_e = puzzle.solution[y, x]
                real_p, real_e = sol[y, x]
                if guess_p != real_p:
                    wrong += 1
                elif guess_e != real_e:
                    wrong_rot += 1
                else:
                    correct += 1

        total = h * w

        print(f"Wrong rotation: {wrong_rot} -> {wrong_rot / total * 100:.2f}%")
        print(f"Wrong piece: {wrong} -> {wrong / total * 100:.2f}%")
        print(f"Correct: {correct} -> {correct / total * 100:.2f}%")

    puzzle_solver.puzzle.show_solution()
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()

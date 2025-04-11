import time
from typing import Optional

import cv2
import numpy as np

from edge import Edge, EdgeType
from piece import Piece, PieceType
from puzzle import Puzzle

PATH = "res/24_pieces_sweater_kevin_small.jpg"
#PATH = "res/49_pieces_sweater_kevin_small2.jpg"
#PATH = "res/49.jpg"

OFFSETS = [
    ( 0, -1),
    (-1,  0),
    ( 0,  1),
    ( 1,  0),
]

def find_pieces_matches(pieces: list[Piece], debug: bool = True):
    """result = [
        [
            (-1, -1) for _ in range(len(pieces))
        ]
        for _ in range(len(pieces))
    ]"""
    result: np.array = np.full((len(pieces), len(pieces), 2), -1)

    cornerPiece: Piece = None
    # find first corner
    for p in pieces:
        if p.type == PieceType.CORNER:
            cornerPiece = p
            break
    if not cornerPiece:
        if debug:
            print("no first corner found")
        else:
            print("first corner found", cornerPiece)

    def rotate(dx: int, dy: int) -> tuple[int, int]:
        return -dy, dx

    def get_edges_at(x, y) -> list[Optional[Edge]]:
        edges: list[Optional[Edge]] = []
        for dx, dy in OFFSETS:
            x2 = x + dx
            y2 = y + dy
            edge = None
            if 0 <= x2 < result.shape[1] and 0 <= y2 < result.shape[0]:
                piece_id, top_idx = result[y2, x2]
                if piece_id != -1:
                    piece = pieces[piece_id]
                    edge = piece.get_edge_by_offset(top_idx, -dx, -dy)

            edges.append(edge)

        return edges

    def createBorderMatches(
            remainingPieces: list[Piece],
            x = 0,
            y = 0,
            dx = 0,
            dy = 0,
            corner_placed = 0)-> bool:

        #print(f"Remaining: {len(remainingPieces)} / {len(pieces)} / {x=}, {y=}, {dx=}, {dy=}, lastPiece={result[y - dy, x - dx]}")
        if len(remainingPieces) == 0:
            #print("  Weird, remainingPieces is empty !")
            return False

        currentPiece: Piece = pieces[result[y - dy, x - dx, 0]]
        currentEdge = currentPiece.get_edge_by_offset(result[y - dy, x - dx, 1], dx, dy)

        prevEdge = currentEdge.get_prev()
        nextEdge = currentEdge.get_next()
        prevEdgeIsFlat = prevEdge.type == EdgeType.FLAT
        nextEdgeIsFlat = nextEdge.type == EdgeType.FLAT

        neighborEdges = get_edges_at(x, y)


        matchValues = []
        for p in remainingPieces:

            if p.type ==  PieceType.CORNER:
                if corner_placed == 3 and x != 0:
                    continue

                if corner_placed == 2:
                    area = (x + 1) * (y + 1)
                    if area != len(pieces):
                        continue

            elif corner_placed == 2:
                area = (x + 1) * (y + 1)
                if area >= len(pieces):
                    continue

            for i, e in enumerate(p.edges):
                if prevEdgeIsFlat != (e.get_next().type == EdgeType.FLAT):
                    continue

                if nextEdgeIsFlat != (e.get_prev().type == EdgeType.FLAT):
                    continue

                # ----

                matchValue = 0

                # neighborEdges = [top, left, bottom, right]
                #
                # i = 0 / dx=1 / left
                target_i = 0
                if dx == 1:
                    target_i = 1
                elif dy == -1:
                    target_i = 2
                elif dx == -1:
                    target_i = 3
                offset = (i - target_i)
                # target_i = 1
                # offset = 2

                for j, ne in enumerate(neighborEdges):
                    edgeToCheck = p.edges[(j + offset) % 4]

                    if ne is None:
                        continue

                    matchValue += edgeToCheck.matches_with(ne)

                if matchValue != np.inf:
                    matchValues.append((e, matchValue))

        matchValues = sorted(matchValues, key=lambda m: m[1])
        #print(f"  {len(matchValues)} possible pieces")
        for edge, val in matchValues:
            top_id = edge.idx
            if dx == 1:
                top_id -= 1
            elif dx == -1:
                top_id += 1
            elif dy == -1:
                top_id += 2
            top_id %= 4

            new_match = (edge.piece.id, top_id)
            #print(f"    Placing piece {new_match} ({val=})")
            remainingPieces2 = list(
                filter(
                    lambda p: p.id != edge.piece.id,
                    remainingPieces
                )
            )

            dx2, dy2 = dx, dy
            if edge.piece.type == PieceType.CORNER:
                #print("    Corner placed, turning right")
                dx2, dy2 = rotate(dx, dy)

            x2 = x + dx2
            y2 = y + dy2

            if y2 < 0 or y2 >= result.shape[0] or x2 < 0 or x2 >= result.shape[1]:
                #print("    Arrived at edge ! Testing another piece")
                continue

            if result[y2, x2, 0] != -1:
                #print(f"    There's already a piece here ({result[y2, x2]}), turning right")
                dx2, dy2 = rotate(dx2, dy2)

            x2 = x + dx2
            y2 = y + dy2

            if y2 < 0 or y2 >= result.shape[0] or x2 < 0 or x2 >= result.shape[1]:
                #print("    Arrived at edge ! (2) Testing another piece")
                continue

            if result[y2, x2, 0] != -1:
                #print("    Reached a dead-end")
                if len(remainingPieces2) == 0:
                    #print("    All pieces have been placed, found a solution")
                    result[y, x] = new_match
                    return True

                #print("    Some pieces left, testing another piece")

                if edge.piece.type == PieceType.CORNER:
                    continue

                return False

            result[y, x] = new_match
            if createBorderMatches(remainingPieces2, x2, y2, dx2, dy2, corner_placed + 1 if edge.piece.type == PieceType.CORNER else corner_placed):
                return True

            result[y, x] = [-1, -1]
            #print(f"    Removing {(x, y)} = {result[y, x]}, testing another piece")

        #print("  No solution found, backtracking")
        return False



    flat_indices = []
    for i, e in enumerate(cornerPiece.edges):
        if e.type == EdgeType.FLAT:
            flat_indices.append(i)
    flat_indices = tuple(sorted(flat_indices))
    top_index = 3 if flat_indices == (0, 3) else flat_indices[0]

    result[0, 0] = [cornerPiece.id, top_index]
    remainingPieces = list(
        filter(
            lambda p: p.id != cornerPiece.id,
            pieces
        )
    )

    t0 = time.time()
    solved = createBorderMatches(remainingPieces, x=1, dx=1, corner_placed=1)
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")
    if not solved:
        print("Could not find a solution !")
        return None
    return result


def main():
    print("##########################################")
    print("##       Fantastic  Puzzle  Solver      ##")
    print("## by A. Gomes, L. Heredero & K. Voisin ##")
    print("##########################################")
    print()

    img = cv2.imread(PATH)
    imgs = [
        cv2.imread("res/49_pieces_eagles/1.jpg"),
        cv2.imread("res/49_pieces_eagles/2.jpg"),
        cv2.imread("res/49_pieces_eagles/3.jpg"),
    ]

    puzzle: Puzzle = Puzzle.from_image(img)
    puzzle: Puzzle = Puzzle.from_images(imgs)
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

    puzzle.solution = find_pieces_matches(puzzle.pieces)
    coords = np.argwhere(puzzle.solution[..., 0] != -1)
    y, x, h, w = cv2.boundingRect(coords)
    #np.savetxt("/tmp/sol.txt", puzzle.solution[..., 0], fmt="%4d")
    puzzle.solution = s = puzzle.solution[y:y+h,x:x+w]
    print(s.shape)
    print(s[..., 0])

    sol = solution49
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
    puzzle.show_solution()
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()

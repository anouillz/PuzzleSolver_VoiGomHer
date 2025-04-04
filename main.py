import cv2
from matplotlib import pyplot as plt

from contours import find_piece_contours
from corners import find_corners
from piece import Piece,PieceType
from edge import Edge,EdgeType
import numpy as np

from puzzle import Puzzle

PATH = "res/24_pieces_sweater_kevin_small.jpg"
#PATH = "res/49_pieces_sweater_kevin_small2.jpg"
#PATH = "res/49.jpg"

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

    def createBorderMatches(
            remainingPieces: list[Piece],
            x = 0,
            y = 0,
            dx = 0,
            dy = 0,
            corner_placed = 0)-> bool:

        print(f"Remaining: {len(remainingPieces)} / {len(pieces)} / {x=}, {y=}, {dx=}, {dy=}, lastPiece={result[y - dy, x - dx]}")
        if len(remainingPieces) == 0:
            print("  Weird, remainingPieces is empty !")
            return False

        currentPiece: Piece = pieces[result[y - dy, x - dx, 0]]
        currentEdge = currentPiece.get_edge_by_offset(result[y - dy, x - dx, 1], dx, dy)

        prevEdge = currentEdge.get_prev()
        nextEdge = currentEdge.get_next()
        prevEdgeIsFlat = prevEdge.type == EdgeType.FLAT
        nextEdgeIsFlat = nextEdge.type == EdgeType.FLAT


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

                matchValue = currentEdge.matches_with(e)
                if matchValue != np.inf:
                    matchValues.append((e, matchValue))

        matchValues = sorted(matchValues, key=lambda m: m[1])
        print(f"  {len(matchValues)} possible pieces")
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
            print(f"    Placing piece {new_match} ({val=})")
            remainingPieces2 = list(
                filter(
                    lambda p: p.id != edge.piece.id,
                    remainingPieces
                )
            )

            dx2, dy2 = dx, dy
            if edge.piece.type == PieceType.CORNER:
                print("    Corner placed, turning right")
                dx2, dy2 = rotate(dx, dy)

            x2 = x + dx2
            y2 = y + dy2

            if y2 < 0 or y2 >= result.shape[0] or x2 < 0 or x2 >= result.shape[1]:
                print("    Arrived at edge ! Testing another piece")
                continue

            if result[y2, x2, 0] != -1:
                print(f"    There's already a piece here ({result[y2, x2]}), turning right")
                dx2, dy2 = rotate(dx2, dy2)

            x2 = x + dx2
            y2 = y + dy2

            if y2 < 0 or y2 >= result.shape[0] or x2 < 0 or x2 >= result.shape[1]:
                print("    Arrived at edge ! (2) Testing another piece")
                continue

            if result[y2, x2, 0] != -1:
                print("    Reached a dead-end")
                if len(remainingPieces2) == 0:
                    print("    All pieces have been placed, found a solution")
                    result[y, x] = new_match
                    return True

                print("    Some pieces left, testing another piece")

                if edge.piece.type == PieceType.CORNER:
                    continue

                return False



            result[y, x] = new_match
            if createBorderMatches(remainingPieces2, x2, y2, dx2, dy2, corner_placed + 1 if edge.piece.type == PieceType.CORNER else corner_placed):
                return True

            result[y, x] = [-1, -1]
            print(f"    Removing {(x, y)} = {result[y, x]}, testing another piece")

        print("  No solution found, backtracking")
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

    if not createBorderMatches(remainingPieces, x=1, dx=1, corner_placed=1):
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

    puzzle: Puzzle = Puzzle.from_image(img)
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
    print(puzzle)
    for piece in puzzle.pieces:
        #cv2.imshow("a", piece.get_rotated_img(0))
        piece.draw(img)
        #cv2.waitKey(-1)

    cv2.imshow("Pieces", img)

    puzzle.solution = find_pieces_matches(puzzle.pieces)
    coords = np.argwhere(puzzle.solution[..., 0] != -1)
    y, x, h, w = cv2.boundingRect(coords)
    #np.savetxt("/tmp/sol.txt", puzzle.solution[..., 0], fmt="%4d")
    puzzle.solution = s = puzzle.solution[y:y+h,x:x+w]
    print(s.shape)
    print(s[..., 0])

    puzzle.show_solution()
    cv2.waitKey(-1)



if __name__ == '__main__':
    main()

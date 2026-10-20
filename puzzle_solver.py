import random
import time
from typing import Optional

import cv2
import numpy as np

from edge import Edge, EdgeType
from piece import Piece, PieceType
from puzzle import Puzzle

OFFSETS = [
    (0, -1),
    (-1, 0),
    (0, 1),
    (1, 0),
]


class PuzzleSolver:

    def __init__(self, puzzle):
        self.puzzle: Puzzle = puzzle
        self._best_value = 0
        self._best_solution = None

    def solve(self, return_first: bool = False, debug: bool = False):
        result: np.array = np.full((len(self.puzzle.pieces), len(self.puzzle.pieces), 2), -1)

        corners = []
        # find first corner
        for p in self.puzzle.pieces:
            if p.type == PieceType.CORNER:
                corners.append(p)

        corner_piece: Optional[Piece] = None
        if len(corners) != 0:
            corner_piece = random.choice(corners)

        if corner_piece is None:
            if debug:
                print("no first corner found")
            else:
                print("first corner found", corner_piece)

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
                        piece = self.puzzle.pieces[piece_id]
                        edge = piece.get_edge_by_offset(top_idx, -dx, -dy)

                edges.append(edge)

            return edges

        def createBorderMatches(
                remainingPieces: list[Piece],
                x=0,
                y=0,
                dx=0,
                dy=0,
                cur_w=0,
                cur_h=0,
                tr_placed=False,
                bl_placed=False,
                value=0) -> bool:

            found_sol = False

            # print(f"Remaining: {len(remainingPieces)} / {len(self.pieces)} / {x=}, {y=}, {dx=}, {dy=}, {cur_w=}, {cur_h=}, lastPiece={result[y - dy, x - dx]}")
            # print(result[..., 0])
            # input()
            # self.solution = result
            # self.show_solution()
            # if cv2.waitKey(100) != -1:
            #    cv2.waitKey(-1)

            depth = len(self.puzzle.pieces) - len(remainingPieces)
            if depth > self.puzzle.max_depth:
                print(f"New max depth: {depth}")
                self.puzzle.solution = result
                self.puzzle.show_solution(True, True)
                cv2.waitKey(100)
                self.puzzle.max_depth = depth

            if len(remainingPieces) == 0:
                # print("  Weird, remainingPieces is empty !")
                return False

            neighborEdges = get_edges_at(x, y)

            matchValues = []
            for p in remainingPieces:
                if False and p.type == PieceType.CORNER:
                    if tr_placed and bl_placed:
                        if x != cur_w - 1 or y != cur_h - 1:
                            continue

                        area = cur_w * cur_h
                        if area != len(self.puzzle.pieces):
                            continue

                    else:
                        if dx == 1 and x != cur_w:
                            continue

                        if dy == 1 and y != cur_h:
                            continue

                for i, e in enumerate(p.edges):

                    matchValue = 0

                    # neighborEdges = [top, left, bottom, right]
                    target_i = 0
                    if dx == 1:
                        target_i = 1
                    elif dy == -1:
                        target_i = 2
                    elif dx == -1:
                        target_i = 3
                    offset = (i - target_i)

                    can_be_placed: bool = True
                    for j, ne in enumerate(neighborEdges):
                        edgeToCheck = p.edges[(j + offset) % 4]

                        prevEdge = edgeToCheck.get_prev()
                        nextEdge = edgeToCheck.get_next()
                        prevEdgeIsFlat = prevEdge.type == EdgeType.FLAT
                        nextEdgeIsFlat = nextEdge.type == EdgeType.FLAT

                        if ne is not None and prevEdgeIsFlat != (ne.get_next().type == EdgeType.FLAT):
                            can_be_placed = False
                            break

                        if ne is not None and nextEdgeIsFlat != (ne.get_prev().type == EdgeType.FLAT):
                            can_be_placed = False
                            break

                        if ne is None:
                            continue

                        matchValue += edgeToCheck.matches_with(ne)

                    if not can_be_placed:
                        continue

                    if matchValue != np.inf:
                        matchValues.append((e, matchValue))

            matchValues = sorted(matchValues, key=lambda m: m[1])
            # print(f"  {len(matchValues)} possible pieces")
            for edge, val in matchValues:
                value2 = value + val
                if self._best_solution is not None and value2 >= self._best_value:
                    print("Value too big, cutting branch")
                    continue
                top_id = edge.idx
                if dx == 1:
                    top_id -= 1
                elif dx == -1:
                    top_id += 1
                elif dy == -1:
                    top_id += 2
                top_id %= 4

                new_match = (edge.piece.id, top_id)
                # print(f"    Placing piece {new_match} ({val=})")
                remainingPieces2 = list(
                    filter(
                        lambda p: p.id != edge.piece.id,
                        remainingPieces
                    )
                )

                tr_placed2 = tr_placed
                bl_placed2 = bl_placed

                dx2 = dx
                dy2 = dy

                if x == 0:
                    dx2, dy2 = 1, 0

                elif y == 0:
                    dx2, dy2 = 0, 1

                x2 = x + dx2
                y2 = y + dy2

                cur_w2 = max(cur_w, x + 1)
                cur_h2 = max(cur_h, y + 1)

                if edge.piece.type == PieceType.CORNER:
                    # print("Placing corner")
                    if x == 0:
                        # print("Placing bottom-left corner")
                        bl_placed2 = True
                    elif y == 0:
                        # print("Placing top-right corner")
                        tr_placed2 = True
                # →→→→→→
                if dx == 1:
                    if x == cur_w - 1:
                        if tr_placed:
                            x2 = 0
                            y2 = y + 1
                        else:
                            x2 = x + 1
                            y2 = 0

                elif dy == 1:
                    if y == cur_h - 1:
                        if bl_placed:
                            x2 = x + 1
                            y2 = 0
                        else:
                            x2 = 0
                            y2 = y + 1

                if x == 1 and y == 0:
                    x2, y2 = 0, 1

                if len(remainingPieces2) == 0:
                    result[y, x] = new_match
                    self._best_value = value
                    self._best_solution = np.copy(result)
                    print(f"New best solution: {value=}")
                    self.puzzle.solution = self._best_solution
                    self.puzzle.show_solution(show=True, save=True)
                    input("Press ENTER to continue")
                    return True

                result[y, x] = new_match
                if createBorderMatches(
                        remainingPieces2,
                        x2, y2,
                        dx2, dy2,
                        cur_w2, cur_h2,
                        tr_placed2,
                        bl_placed2,
                        value2):
                    found_sol = True
                    if return_first:
                        return True

                result[y, x] = [-1, -1]
                # print(f"    Removing {(x, y)} = {result[y, x]}, testing another piece")

            # print("  No solution found, backtracking")
            return found_sol

        flat_indices = []
        for i, e in enumerate(corner_piece.edges):
            if e.type == EdgeType.FLAT:
                flat_indices.append(i)
        flat_indices = tuple(sorted(flat_indices))
        top_index = 3 if flat_indices == (0, 3) else flat_indices[0]

        result[0, 0] = [corner_piece.id, top_index]
        remainingPieces = list(
            filter(
                lambda p: p.id != corner_piece.id,
                self.puzzle.pieces
            )
        )

        t0 = time.time()
        solved = createBorderMatches(remainingPieces, x=1, dx=1, cur_w=1, cur_h=1)
        t1 = time.time()
        print(f"Time taken: {t1 - t0:.2f} seconds")
        if not solved:
            print("Could not find a solution !")
            return None

        self.puzzle.solution = self._best_solution
        return self._best_solution
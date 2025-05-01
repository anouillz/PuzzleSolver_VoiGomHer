from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from contours import find_piece_contours
from corners import find_corners
from edge import Edge, EdgeType
from piece import Piece, PieceType


OFFSETS = [
    ( 0, -1),
    (-1,  0),
    ( 0,  1),
    ( 1,  0),
]


class Puzzle:
    def __init__(self, pieces: list[Piece], img: Optional[np.array] = None):
        self.img: Optional[np.array] = img
        self.pieces: list[Piece] = pieces
        self.solution: Optional[np.array] = None
        self.sol_i = 0

    def __repr__(self):
        return f"<Puzzle {len(self.pieces)} pieces - {'UNSOLVED' if self.solution is None else 'SOLVED'}>"

    @staticmethod
    def from_image(img: np.array, img2: np.array, img3: np.array, img4: np.array) -> Puzzle:
        # Contours
        print("Finding contours")
        piece_contours = find_piece_contours(img, img2, img3, img4)
        n_pieces = len(piece_contours)
        print(f"{n_pieces} pieces")

        # Corners + edges
        print("Identifying corners and edges")
        pieces = []
        img2 = img.copy()
        for i, (contour, box) in enumerate(piece_contours):
            corners, edges, edge_contours = find_corners(contour)
            piece = Piece(i, contour, box, corners, edges, edge_contours, img)
            pieces.append(piece)
            piece.draw(img2)

        cv2.imwrite(f"/tmp/img.jpg", img2)
        return Puzzle(pieces, img)

    @staticmethod
    def from_images(imgs: list[np.array]) -> Puzzle:
        pieces = []
        piece_id = 0
        for i, img in enumerate(imgs):
            print(f"Processing image {i+1}/{len(imgs)}")

            # Contours
            print("Finding contours")
            piece_contours = find_piece_contours(img)
            n_pieces = len(piece_contours)
            print(f"{n_pieces} pieces")
            img2 = img.copy()

            # Corners + edges
            print("Identifying corners and edges")
            for contour, box in piece_contours:
                corners, edges, edge_contours = find_corners(contour)
                piece = Piece(piece_id, contour, box, corners, edges, edge_contours, img)
                pieces.append(piece)
                piece.draw(img2)
                piece_id += 1

            cv2.imwrite(f"/tmp/img_{i}.jpg", img2)
            cv2.waitKey(-1)

        return Puzzle(pieces, None)

    def solve(self, return_first: bool = False, debug: bool = False):
        result: np.array = np.full((len(self.pieces), len(self.pieces), 2), -1)

        corner_piece: Optional[Piece] = None
        # find first corner
        for p in self.pieces:
            if p.type == PieceType.CORNER:
                corner_piece = p
                break

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
                        piece = self.pieces[piece_id]
                        edge = piece.get_edge_by_offset(top_idx, -dx, -dy)

                edges.append(edge)

            return edges

        self._best_value = 0
        self._best_solution = None

        def createBorderMatches(
                remainingPieces: list[Piece],
                x = 0,
                y = 0,
                dx = 0,
                dy = 0,
                corner_placed = 0,
                value = 0)-> bool:

            found_sol = False
            #print(f"Remaining: {len(remainingPieces)} / {len(pieces)} / {x=}, {y=}, {dx=}, {dy=}, lastPiece={result[y - dy, x - dx]}")
            if len(remainingPieces) == 0:
                #print("  Weird, remainingPieces is empty !")
                return False

            currentPiece: Piece = self.pieces[result[y - dy, x - dx, 0]]
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
                        if area != len(self.pieces):
                            continue

                elif corner_placed == 2:
                    area = (x + 1) * (y + 1)
                    if area >= len(self.pieces):
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
                        self._best_value = value
                        self._best_solution = np.copy(result)
                        print(f"New best solution: {value=}")
                        self.solution = self._best_solution
                        self.show_solution(show=True, save=True)
                        input("Press ENTER to continue")
                        return True

                    #print("    Some pieces left, testing another piece")

                    if edge.piece.type == PieceType.CORNER:
                        continue

                    return False

                result[y, x] = new_match
                if createBorderMatches(
                    remainingPieces2,
                    x2, y2,
                    dx2, dy2,
                    corner_placed + 1 if edge.piece.type == PieceType.CORNER else corner_placed,
                    value2):
                    found_sol = True
                    if return_first:
                        return True

                result[y, x] = [-1, -1]
                #print(f"    Removing {(x, y)} = {result[y, x]}, testing another piece")

            print("  No solution found, backtracking")
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
                self.pieces
            )
        )

        t0 = time.time()
        solved = createBorderMatches(remainingPieces, x=1, dx=1, corner_placed=1)
        t1 = time.time()
        print(f"Time taken: {t1 - t0:.2f} seconds")
        if not solved:
            print("Could not find a solution !")
            return None

        self.solution = self._best_solution
        return self._best_solution

    def show_solution(self, show: bool = True, save: bool = False):
        if self.solution is None:
            raise ValueError("Cannot show solution because puzzle is not solved. Try to call the solve() method first")

        coords = np.argwhere(self.solution[..., 0] != -1)
        y, x, h, w = cv2.boundingRect(coords)
        #np.savetxt("/tmp/sol.txt", self.solution[..., 0], fmt="%4d")
        self.solution = self.solution[y:y+h,x:x+w]
        print(self.solution.shape)
        print(self.solution[..., 0])

        img: np.array = np.zeros((0, 0, 4), np.uint8)
        oy = 0
        max_w = 0
        for y in range(self.solution.shape[0]):
            height = 0
            ox = 0
            for x in range(self.solution.shape[1]):
                idx, rot = self.solution[y, x]
                if idx == -1:
                    ox += max_w
                    continue
                piece: Piece = self.pieces[idx]
                piece_img: np.array = piece.get_rotated_img(rot)
                mask = piece_img[..., 3] != 0
                x2 = ox + piece_img.shape[1]
                y2 = oy + piece_img.shape[0]
                if x2 >= img.shape[1] or y2 >= img.shape[0]:
                    img2 = np.zeros((max(img.shape[0], y2), max(img.shape[1], x2), 4), np.uint8)
                    img2[:img.shape[0], :img.shape[1]] = img
                    img = img2
                img[oy:y2, ox:x2][mask] = piece_img[mask]
                height = max(height, piece_img.shape[0])
                ox += piece_img.shape[1]
                max_w = max(max_w, piece_img.shape[1])
                #cv2.imshow("Solution", img)
                #cv2.waitKey(100)
            oy += height

        if show:
            cv2.imshow("Solution", img)
            cv2.waitKey(100)
        if save:
            cv2.imwrite(f"/tmp/solution_{self.sol_i}.jpg", img)
            self.sol_i += 1
import random
import time
from typing import Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt

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
    def __init__(self, puzzle: Puzzle, debug: bool = False):
        self.puzzle: Puzzle = puzzle
        self.debug: bool = debug
        self._best_value = 0
        self._best_solution = None
        self.result: np.array = np.zeros((0, 0, 2))
        self.remaining_pieces: set[Piece] = set(self.puzzle.pieces)
        self.next_positions: set[tuple[int, int]] = set()
        self.max_depth: int = 0

    def find_corner(self) -> Optional[Piece]:
        corners = []

        for p in self.puzzle.pieces:
            if p.type == PieceType.CORNER:
                corners.append(p)

        if len(corners) != 0:
            return random.choice(corners)
        return None

    def place_corner(self):
        corner_piece = self.find_corner()
        if corner_piece is None:
            if self.debug:
                print("no first corner found")
            else:
                print("first corner found", corner_piece)

        flat_indices = []
        for i, e in enumerate(corner_piece.edges):
            if e.type == EdgeType.FLAT:
                flat_indices.append(i)
        flat_indices = tuple(sorted(flat_indices))
        top_index = 3 if flat_indices == (0, 3) else flat_indices[0]

        self.result[0, 0] = [corner_piece.id, top_index]
        self.remaining_pieces.remove(corner_piece)

        self.next_positions = {
            (0, 1),
            (1, 0)
        }

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.result.shape[1] and 0 <= y < self.result.shape[0]

    def get_edges_at(self, x: int, y: int) -> list[Optional[Edge]]:
        edges: list[Optional[Edge]] = []
        for dx, dy in OFFSETS:
            x2: int = x + dx
            y2: int = y + dy
            edge: Optional[Edge] = None

            if self.is_in_bounds(x2, y2):
                piece_id, top_idx = self.result[y2, x2]
                if piece_id != -1:
                    piece: Piece = self.puzzle.pieces[piece_id]
                    edge: Edge = piece.get_edge_by_offset(top_idx, -dx, -dy)

            edges.append(edge)

        return edges

    def get_candidates(self, x: int, y: int) -> list[tuple[Edge, float]]:
        neighbor_edges: list[Optional[Edge]] = self.get_edges_at(x, y)

        match_values: list[tuple[Edge, float]] = []
        for piece in self.remaining_pieces:
            for i, e in enumerate(piece.edges):
                match_value = 0

                # neighbor_edges = [top, left, bottom, right]

                can_be_placed: bool = True
                for j, ne in enumerate(neighbor_edges):
                    edge_to_check = piece.edges[(j + i) % 4]

                    prev_edge = edge_to_check.get_prev()
                    next_edge = edge_to_check.get_next()
                    prev_edge_is_flat = prev_edge.type == EdgeType.FLAT
                    next_edge_is_flat = next_edge.type == EdgeType.FLAT

                    if ne is not None and prev_edge_is_flat != (ne.get_next().type == EdgeType.FLAT):
                        can_be_placed = False
                        break

                    if ne is not None and next_edge_is_flat != (ne.get_prev().type == EdgeType.FLAT):
                        can_be_placed = False
                        break

                    if ne is None:
                        continue

                    match_value += edge_to_check.matches_with(ne)

                if not can_be_placed:
                    continue

                if match_value != np.inf:
                    match_values.append((e, match_value))

        match_values = sorted(match_values, key=lambda m: m[1])
        return match_values

    def place_piece(self, edge: Edge, value: float, pos: tuple[int, int]):
        self.result[pos[1], pos[0]] = (edge.piece.id, edge.idx)
        self.next_positions.remove(pos)
        self.add_positions(*pos)
        self.remaining_pieces.remove(edge.piece)

    def add_positions(self, x: int, y: int):
        for dx, dy in OFFSETS:
            x2, y2 = x + dx, y + dy
            if self.is_in_bounds(x2, y2):
                if self.result[y2, x2][0] == -1:
                    self.next_positions.add((x2, y2))

    def remove_piece(self, pos: tuple[int, int]):
        piece_id, top_idx = self.result[pos[1], pos[0]]
        self.result[pos[1], pos[0]] = (-1, -1)
        self.remove_positions(*pos)
        self.next_positions.add(pos)
        self.remaining_pieces.add(self.puzzle.pieces[piece_id])

    def remove_positions(self, x: int, y: int):
        for dx, dy in OFFSETS:
            x2, y2 = x + dx, y + dy
            if self.is_in_bounds(x2, y2):
                has_nb: bool = False
                for dx2, dy2 in OFFSETS:
                    x3, y3 = x2 + dx2, y2 + dy2
                    if self.is_in_bounds(x3, y3) and self.result[y3, x3][0] != -1:
                        has_nb = True
                        break

                if not has_nb:
                    self.next_positions.discard((x2, y2))

    def backtrack(self):
        worst_pos: tuple[int, int] = (0, 0)
        worst_value: float = -1

        for y in range(self.result.shape[0]):
            for x in range(self.result.shape[1]):
                value: float = self.get_value_at(x, y)
                if value > worst_value:
                    worst_pos = (x, y)
                    worst_value = value

        if worst_value != -1:
            self.remove_piece(worst_pos)

    def get_value_at(self, x: int, y: int) -> float:
        r = self.get_piece_at(x, y)
        if r is None:
            return -1

        piece: Piece = r[0]
        top_idx: int = r[1]

        value: float = 0
        for dx, dy in OFFSETS:
            x2, y2 = x + dx, y + dy
            r2 = self.get_piece_at(x2, y2)
            if r2 is None:
                continue

            piece2: Piece = r2[0]
            top_idx2: int = r2[1]

            edge1: Edge = piece.get_edge_by_offset(top_idx, dx, dy)
            edge2: Edge = piece2.get_edge_by_offset(top_idx2, -dx, -dy)
            value += edge1.matches_with(edge2)

        return value

    def get_piece_at(self, x: int, y: int) -> Optional[tuple[Piece, int]]:
        if not self.is_in_bounds(x, y):
            return None

        piece_id, top_idx = self.result[y, x]
        if piece_id == -1:
            return None

        piece: Piece = self.puzzle.pieces[piece_id]
        return piece, top_idx

    def plot_heatmap(self):
        plt.figure(figsize=(10, 8))
        values = np.zeros(self.result.shape[:2])
        for y in range(self.result.shape[0]):
            for x in range(self.result.shape[1]):
                values[y, x] = self.get_value_at(x, y)

        img = plt.imshow(values)
        plt.colorbar(img)
        plt.show()

    def solve(self, debug: bool = False):
        self.result: np.array = np.full((len(self.puzzle.pieces), len(self.puzzle.pieces), 2), -1)
        self.remaining_pieces = set(self.puzzle.pieces)
        self.max_depth = 0

        self.place_corner()

        t0 = time.time()
        i = 0
        while len(self.remaining_pieces) > 0:
            best_edge: Optional[Edge] = None
            best_value: float = 0
            best_pos: tuple[int, int] = (0, 0)

            for x, y in self.next_positions:
                for edge, value in self.get_candidates(x, y):
                    if best_edge is None or value < best_value:
                        best_edge = edge
                        best_value = value
                        best_pos = (x, y)

            if best_edge is None:
                self.backtrack()
            else:
                self.place_piece(best_edge, best_value, best_pos)
                i += 1

                depth = len(self.puzzle.pieces) - len(self.remaining_pieces)
                if depth > self.max_depth:
                    print(f"New max depth: {depth}")
                    if debug:
                        self.puzzle.solution = self.result
                        self.puzzle.show_solution(True, True)
                        self.plot_heatmap()
                        cv2.waitKey(100)
                    self.max_depth = depth

            if i >= 20:
                i = 0
                self.backtrack()

        t1 = time.time()
        print(f"Time taken: {t1 - t0:.2f} seconds")

        self.puzzle.solution = self.result
        return self.result

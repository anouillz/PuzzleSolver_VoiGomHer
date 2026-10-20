from __future__ import annotations

from math import sqrt
from typing import Optional, TYPE_CHECKING

import numpy as np

from edge import Edge
from piece import Piece

if TYPE_CHECKING:
    from puzzle import Puzzle


class Vec:
    def __init__(self, x: float = 0, y: float = 0):
        self.x: float = x
        self.y: float = y

    def __add__(self, other: Vec) -> Vec:
        return Vec(self.x + other.x, self.y + other.y)

    def __radd__(self, other: Vec) -> Vec:
        return Vec(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec) -> Vec:
        return Vec(self.x - other.x, self.y - other.y)

    def __rsub__(self, other: Vec) -> Vec:
        return Vec(self.x - other.x, self.y - other.y)

    def __mul__(self, other: Vec|float) -> float|Vec:
        if isinstance(other, Vec):
            return self.x * other.x + self.y * other.y
        return Vec(self.x * other, self.y * other)

    def __rmul__(self, other: Vec|float) -> float|Vec:
        if isinstance(other, Vec):
            return self.x * other.x + self.y * other.y
        return Vec(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> Vec:
        return Vec(self.x / other, self.y / other)

    def __neg__(self) -> Vec:
        return Vec(-self.x, -self.y)

    def mag(self) -> float:
        return sqrt(self * self)

    def normalized(self) -> Vec:
        mag: float = self.mag()
        if mag == 0:
            return Vec()
        return self / mag

    def __repr__(self) -> str:
        return f"Vec({self.x}, {self.y})"


class Node:
    def __init__(self, pos: Optional[Vec] = None, piece_id: int = 0, corner_idx: int = 0):
        self.pos: Vec = Vec() if pos is None else pos
        self.vel: Vec = Vec()
        self.acc: Vec = Vec()
        self.acc2 = self.acc
        self.piece_id: int = piece_id
        self.corner_idx: int = corner_idx

    def update(self):
        self.vel *= 0.95
        self.vel += self.acc
        self.pos += self.vel
        self.acc2 = self.acc
        self.acc = Vec()

    def get_energy(self) -> float:
         return 0.5 * self.vel * self.vel


class Spring:
    def __init__(self, node1: Node, node2: Node, length: float, k: float = 0.05):
        self.node1: Node = node1
        self.node2: Node = node2
        self.length: float = length
        self.k: float = k

    def update(self):
        vec: Vec = self.node2.pos - self.node1.pos
        dist: float = vec.mag()
        dx: float = self.length - dist
        acc: Vec = (-self.k * dx / dist) * vec
        self.node1.acc += acc
        self.node2.acc -= acc

    def get_energy(self) -> float:
        vec: Vec = self.node2.pos - self.node1.pos
        return 0.5 * self.k * vec * vec


class Mesh:
    def __init__(self):
        self.nodes: list[Node] = []
        self.node_grid: list[list[Node]] = []
        self.springs: list[Spring] = []
        self.t = 1

    def update(self):
        for spring in self.springs:
            spring.update()

        center: Vec = Vec()
        for node in self.nodes:
            center += node.pos
        center /= len(self.nodes)

        for node in self.nodes:
            d: Vec = node.pos - center
            m = d.mag()
            if m == 0:
                continue
            #node.acc += d / m / m * 500
            # Gravitate towards center (0, 0)
            if self.t < 200:
                node.acc -= node.pos.normalized() * 0.01# / 5000
            node.update()

    @staticmethod
    def from_puzzle(puzzle: Puzzle) -> Mesh:
        if puzzle.solution is None:
            raise ValueError("Cannot show solution because puzzle is not solved. Try to call the solve() method first")

        mesh: Mesh = Mesh()
        sol: np.array = puzzle.solution

        h, w, _ = sol.shape
        nodes: list[list[Node]] = []
        for y in range(h * 2):
            row: list[Node] = []
            for x in range(w * 2):
                node: Node = Node(
                    Vec(
                        (x - w) * 100,
                        (y - h) * 100
                    )
                )
                row.append(node)
                mesh.nodes.append(node)
            nodes.append(row)

        edge_offsets: list[tuple[int, int]] = [
            ( 0, -1),
            (-1,  0),
            ( 0,  1),
            ( 1,  0),
        ]
        node_offsets: list[tuple[tuple[int, int], tuple[int, int]]] = [
            ((1, 0), (0, 0)),
            ((0, 0), (0, 1)),
            ((0, 1), (1, 1)),
            ((1, 1), (1, 0)),
        ]

        for y in range(h):
            for x in range(w):
                pid, top_idx = sol[y, x]
                piece: Piece = puzzle.pieces[pid]

                corners: list[Vec] = [
                    Vec(*piece.corners_pos[(i - top_idx) % 4])
                    for i in range(4)
                ]
                l02 = (corners[0] - corners[2]).mag()
                l13 = (corners[1] - corners[3]).mag()
                diag_spring1: Spring = Spring(
                    nodes[y * 2][x * 2],
                    nodes[y * 2 + 1][x * 2 + 1],
                    l13,
                    0.2
                )
                diag_spring2: Spring = Spring(
                    nodes[y * 2 + 1][x * 2],
                    nodes[y * 2][x * 2 + 1],
                    l02,
                    0.2
                )
                mesh.springs.append(diag_spring1)
                mesh.springs.append(diag_spring2)

                nodes[y * 2][x * 2].piece_id = piece.id
                nodes[y * 2][x * 2].corner_idx = (0 - top_idx) % 4
                nodes[y * 2 + 1][x * 2].piece_id = piece.id
                nodes[y * 2 + 1][x * 2].corner_idx = (1 - top_idx) % 4
                nodes[y * 2 + 1][x * 2 + 1].piece_id = piece.id
                nodes[y * 2 + 1][x * 2 + 1].corner_idx = (2 - top_idx) % 4
                nodes[y * 2][x * 2 + 1].piece_id = piece.id
                nodes[y * 2][x * 2 + 1].corner_idx = (3 - top_idx) % 4

                for i in range(4):
                    edx, edy = edge_offsets[i]
                    (dx1, dy1), (dx2, dy2) = node_offsets[i]

                    edge: Edge = piece.get_edge_by_offset(top_idx, edx, edy)

                    node1: Node = nodes[y * 2 + dy1][x * 2 + dx1]
                    node2: Node = nodes[y * 2 + dy2][x * 2 + dx2]

                    edge_spring: Spring = Spring(node1, node2, edge.length, 0.05)
                    mesh.springs.append(edge_spring)

                    x2, y2 = x + edx, y + edy
                    if x2 < 0 or x2 >= w or y2 < 0 or y2 >= h:
                        continue

                    nb_node1: Node = nodes[y * 2 + dy1 + edy][x * 2 + dx1 + edx]
                    nb_node2: Node = nodes[y * 2 + dy2 + edy][x * 2 + dx2 + edx]

                    nb_spring1: Spring = Spring(node1, nb_node1, 0)
                    nb_spring2: Spring = Spring(node2, nb_node2, 0)
                    mesh.springs.append(nb_spring1)
                    mesh.springs.append(nb_spring2)

        mesh.node_grid = nodes
        return mesh

    def render(self, win, energy):
        import pygame

        win.fill(0)
        off = Vec(400, 400)
        for spring in self.springs:
            p1 = spring.node1.pos / 5 + off
            p2 = spring.node2.pos / 5 + off
            pygame.draw.line(win, (255, 255, 255), [p1.x, p1.y], [p2.x, p2.y])
        for node in self.nodes:
            p = node.pos / 5 + off
            pygame.draw.circle(win, (255, 0, 0), [p.x, p.y], 5)
            #pygame.draw.line(win, (0, 255, 0), [p.x, p.y], [p.x + node.acc2.x * 100, p.y + node.acc2.y * 100])

        if len(energy) >= 2:
            pygame.draw.lines(win, (0, 255, 255), False, [
                (i, float(800 - e / 1000000 * 400))
                for i, e in enumerate(energy)
            ])

        pygame.display.flip()

    def animate(self):
        import pygame
        pygame.init()
        win = pygame.display.set_mode([800, 800])
        clock = pygame.time.Clock()
        energy = []
        self.t = 0


        while True:
            self.update()
            self.t += 1
            self.render(win)

    def get_energy(self):
        energy: float = 0
        for node in self.nodes:
            energy += node.get_energy()

        for spring in self.springs:
            energy += spring.get_energy()

        return energy

    def stabilize(self, max_steps: int, animate: bool = False):
        if animate:
            import pygame
            win = pygame.display.set_mode([800, 800])
            clock = pygame.time.Clock()

        step = 0
        energy = np.zeros((20,), dtype=np.float32)

        for i in range(20):
            self.update()
            energy[i] = self.get_energy()

        ref_amplitude = np.max(energy) - np.min(energy)
        amplitude = ref_amplitude

        while step < max_steps and (step < 100 or amplitude > ref_amplitude * 0.01):
            energy = np.roll(energy, -1)
            self.update()
            energy[-1] = self.get_energy()
            amplitude = np.max(energy) - np.min(energy)
            step += 1

            if animate:
                self.render(win, energy)
                clock.tick(20)

        return step

    def get_bounds(self) -> tuple[float, float, float, float]:
        min_x, max_x = 0, 0
        min_y, max_y = 0, 0

        for node in self.nodes:
            min_x = min(min_x, node.pos.x)
            max_x = max(max_x, node.pos.x)
            min_y = min(min_y, node.pos.y)
            max_y = max(max_y, node.pos.y)

        return min_x, min_y, max_x - min_x, max_y - min_y


def main():
    mesh: Mesh = Mesh()

    w = 14
    h = 14
    for y in range(h):
        for x in range(w):
            mesh.nodes.append(Node(Vec(x * 50, y * 50)))

    for y in range(h):
        for x in range(w):
            node1: Node = mesh.nodes[y * w + x]
            for dx, dy in ((1, 0), (0, 1)):
                x2, y2 = x + dx, y + dy
                if x2 < w and y2 < h:
                    node2: Node = mesh.nodes[y2 * w + x2]
                    long: bool = True
                    if dx == 1 and x % 2:
                        long = False
                    elif dy == 1 and y % 2:
                        long = False
                    spring: Spring = Spring(node1, node2, 60 if long else 20)
                    mesh.springs.append(spring)

    mesh.animate()


if __name__ == '__main__':
    main()

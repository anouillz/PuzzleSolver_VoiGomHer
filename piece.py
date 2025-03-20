from __future__ import annotations

import cv2
import numpy as np

from edge import Edge, EdgeType


class Piece:
    EDGE_COLORS = {
        EdgeType.INSIDE: (0, 0, 255),
        EdgeType.FLAT: (0, 255, 255),
        EdgeType.OUTSIDE: (0, 255, 0)
    }
    CORNER_COLOR = (255, 0, 255)

    def __init__(
            self,
            id: int,
            contour: np.array,
            bbox: tuple[int, int, int, int],
            corners: list[tuple[float, float]],
            edges: list[EdgeType],
            edge_contours: list[np.ndarray]):
        self.id: int = id
        self.contour: np.array = contour
        self.corners_pos: np.array = corners
        self.edges: list[Edge] = [
            Edge(self, i, cont, typ)
            for i, (cont, typ) in enumerate(zip(edge_contours, edges))
        ]
        self.bbox: tuple[int, int, int, int] = bbox

    def draw(self, img: np.ndarray):
        for edge in self.edges:
            col = self.EDGE_COLORS[edge.type]
            cv2.polylines(img, [edge.contour], False, col, 2)

        for pt in np.int32(self.corners_pos):
            cv2.circle(img, pt, 3, self.CORNER_COLOR, -1)

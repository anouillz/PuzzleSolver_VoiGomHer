from __future__ import annotations
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from utils import resample_polygon

if TYPE_CHECKING:
    from piece import Piece


class EdgeType(IntEnum):
    INSIDE = -1
    FLAT = 0
    OUTSIDE = 1


class Edge:
    SHAPE_VEC_SIZE = 30

    def __init__(self, piece: Piece, idx: int, contour: np.array, type: EdgeType):
        self.piece: Piece = piece
        self.idx: int = idx
        self.contour: np.array = contour
        self.type: EdgeType = type
        self.length: float = np.linalg.norm(self.contour[0] - self.contour[-1])
        self.shape_vec: np.array = self.compute_shape_vec()
        self.color_vec: np.array = self.compute_color_vec()

    def compute_shape_vec(self):
        vect = []
        c1 = self.contour[0][0]
        c2 = self.contour[-1][0]
        line = c2 - c1
        n = np.array([-line[1], line[0]])
        down = np.linalg.norm(n)

        e = resample_polygon(self.contour[:, 0, :], self.SHAPE_VEC_SIZE)

        for i in e:
            v = i - c1
            s = np.dot(v, n)
            vect.append(s / down)

        return np.array(vect)

    def compute_color_vec(self):
        return np.empty((0,))

    # checks edges length and shape vector
    def matches_with(self, other:Edge):

        if self.type == EdgeType.FLAT or other.type == EdgeType.FLAT:
            return False


        # if length of edges dont match
        if abs(self.length - other.length) / self.length > 0.2:
            return False

        # change other vector direction (car tous les edges sont en clockwise)
        v = other.shape_vec[::-1]

        if self.type != other.type:
            print("----")
            print("P", self.piece.id, " ", self.idx, " ; P", other.piece.id, " ", other.idx)
            print("différence de longueur entre deux edges : ", abs(self.length - other.length) / self.length)
            print("difference de forme: ", np.std(self.shape_vec + v))
            print(self.shape_vec + v)




        return np.linalg.norm(self.shape_vec + v) / self.length < 0.5


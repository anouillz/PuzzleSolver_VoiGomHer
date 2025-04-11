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
    COLOR_VEC_SIZE = 50
    COLOR_RECTANGLE_SIZE = 4
    COLOR_SAMPLE_INSET = 7

    def __init__(self, piece: Piece, idx: int, contour: np.array, type: EdgeType,img:np.ndarray):
        self.piece: Piece = piece
        self.idx: int = idx
        self.contour: np.array = contour
        self.type: EdgeType = type
        self.length: float = np.linalg.norm(self.contour[0] - self.contour[-1])
        self.shape_vec: np.array = self.compute_shape_vec()
        self.color_vec: np.array = self.compute_color_vec(img)

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

    def compute_color_vec(self,img:np.ndarray):
        resampled_contour = resample_polygon(self.contour[:, 0, :],self.COLOR_VEC_SIZE)
        color_vec = []
        for i in range(1,len(resampled_contour)-1):
            before = np.array(resampled_contour[i-1])
            now = np.array(resampled_contour[i])
            after = np.array(resampled_contour[i+1])

            tangant_before = (now-before)
            tangant_after = (after-now)
            tangant = (tangant_after+tangant_before)/2
            normale = np.array([-tangant[1],tangant[0]])
            normale = normale / np.linalg.norm(normale)
            taille = self.COLOR_RECTANGLE_SIZE

            now = np.int32(now - normale * self.COLOR_SAMPLE_INSET)
            rect = img[now[0]-taille:now[0] + taille+1,now[1]-taille:now[1]+taille]
            color_mean = np.mean(rect,axis=(0,1))
            color_vec.append(color_mean)

        return np.array(color_vec)
    
    def compare_color_edge_with(self,e:Edge):
        e_reversed = e.color_vec[::-1]
        matches = []
        ratio_each_color = 0.05
        final_ratio = 0.85
        for i, (f1, f2) in enumerate(zip(self.color_vec, e_reversed)):
            lower_bound = [f2[j] * (1 - ratio_each_color) for j in range(3)]
            upper_bound = [f2[j] * (1 + ratio_each_color) for j in range(3)]
            if all(lower_bound[j] <= f1[j] <= upper_bound[j] for j in range(3)):
                matches.append(i)

        if(len(matches) / len(self.color_vec) >= final_ratio):
            print("match !")
        
        return len(matches) / len(self.color_vec)

    # checks edges length and shape vector
    def matches_with(self, other:Edge) -> float:

        if self.type == EdgeType.FLAT or other.type == EdgeType.FLAT:
            return np.inf


        # if length of edges dont match
        if abs(self.length - other.length) / self.length > 0.2:
            return np.inf

        # change other vector direction (car tous les edges sont en clockwise)
        v = other.shape_vec[::-1]

        if self.type == other.type:
            """
            print("----")
            print("P", self.piece.id, " ", self.idx, " ; P", other.piece.id, " ", other.idx)
            print("différence de longueur entre deux edges : ", abs(self.length - other.length) / self.length)
            print("difference de forme: ", np.std(self.shape_vec + v))
            print(self.shape_vec + v)
            """
            return np.inf

        return np.std(self.shape_vec + v)

    def get_prev(self) -> Edge:
        return self.piece.edges[(self.idx - 1) % 4]

    def get_next(self) -> Edge:
        return self.piece.edges[(self.idx + 1) % 4]

    def __repr__(self):
        return f"<Edge {self.idx} in piece {self.piece.id} - {self.type.name}>"

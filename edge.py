from __future__ import annotations

import colorsys
import math
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
    SHAPE_VEC_SIZE = 60
    COLOR_VEC_SIZE = 100
    COLOR_SAMPLE_RATIO = 1/30
    COLOR_SAMPLE_INSET_RATIO = 1 + 1/20
    CACHE = {}

    def __init__(self, piece: Piece, idx: int, contour: np.array, type: EdgeType,img:np.ndarray):
        self.piece: Piece = piece
        self.idx: int = idx
        self.contour: np.array = contour
        self.type: EdgeType = type
        self.length: float = np.linalg.norm(self.contour[0] - self.contour[-1])
        self.shape_vec, self.shape_vec2 = self.compute_shape_vec()
        self.color_vec: np.array = self.compute_color_vec(img)

    def compute_shape_vec(self):
        vect = []
        vect2 = []
        c1 = self.contour[0][0]
        c2 = self.contour[-1][0]
        line = c2 - c1
        n = np.array([-line[1], line[0]])
        norm = np.linalg.norm(n)

        pts = resample_polygon(self.contour[:, 0, :], self.SHAPE_VEC_SIZE)

        for pt in pts:
            v = pt - c1
            vect.append(np.dot(v, n) / norm / norm)
            vect2.append(np.dot(v, line) / norm / norm)

        return np.array(vect), np.array(vect2)

    def compute_color_vec(self,img:np.ndarray):
        c1 = self.contour[0][0]
        c2 = self.contour[-1][0]
        rectangle_size = np.linalg.norm(c2-c1) * self.COLOR_SAMPLE_RATIO

        sample_inset = rectangle_size * self.COLOR_SAMPLE_INSET_RATIO


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
            taille = int(rectangle_size)

            now = np.int32(now - normale * sample_inset)
            rect = img[now[1]-taille:now[1] + taille+1,now[0]-taille:now[0]+taille]
            color_mean = np.mean(rect,axis=(0,1))
            color_vec.append(color_mean)
            #cv2.rectangle(img,(now[0]-taille,now[1]-taille),(now[0] + taille+1,now[1]+taille),color_mean,-1)


        return np.array(color_vec)
    
    def compare_color_edge_with(self,e:Edge):
        e_reversed = e.color_vec[::-1]
        matches = []
        for i, (f1, f2) in enumerate(zip(self.color_vec, e_reversed)):
            r1, g1, b1 = f1
            r2, g2, b2 = f2
            h1, s1, v1 = colorsys.rgb_to_hsv(r1/255,g1/255,b1/255)
            h2, s2, v2 = colorsys.rgb_to_hsv(r2/255,g2/255,b2/255)

            h1,s1,v1 = h1*360,s1*100,v1*100
            h2,s2,v2 = h2*360,s2*100,v2*100

            distance_hue = self.hue_color_distance(h1,s1,v1,h2,s2,v2)
            matches.append(distance_hue)
        max_dist = math.sqrt(180*180 + 100*100*2)

        return np.sum(np.array(matches) / max_dist) / self.COLOR_VEC_SIZE

    def hue_color_distance(self,h1,s1,v1,h2,s2,v2,hue_weight=3,sat_weight=1,val_weight=1) -> float:
        hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
        distance = math.sqrt(
            (hue_weight * hue_diff) ** 2 +
            (sat_weight * (s1 - s2)) ** 2 +
            (val_weight * (v1 - v2)) ** 2
            )
        return distance

    # checks edges length and shape vector
    def matches_with(self, other:Edge) -> float:
        p1, e1, p2, e2 = self.piece.id, self.idx, other.piece.id, other.idx
        if p1 > p2:
            p1, p2, e1, e2 = p2, p1, e2, e1
        id = (p1, e1, p2, e2)
        if id in self.CACHE:
            return self.CACHE[id]

        if self.type == EdgeType.FLAT or other.type == EdgeType.FLAT:
            return np.inf


        # if length of edges dont match
        """if abs(self.length - other.length) / self.length > 0.2:
            return np.inf"""

        # change other vector direction (car tous les edges sont en clockwise)
        v = other.shape_vec[::-1]
        v2 = 1 - other.shape_vec2[::-1]

        if self.type == other.type:
            """
            print("----")
            print("P", self.piece.id, " ", self.idx, " ; P", other.piece.id, " ", other.idx)
            print("différence de longueur entre deux edges : ", abs(self.length - other.length) / self.length)
            print("difference de forme: ", np.std(self.shape_vec + v))
            print(self.shape_vec + v)
            """
            return np.inf

        # return_value = np.std(self.shape_vec+v)
        color_diff = self.compare_color_edge_with(other)
        #return return_value

        #return np.std(self.shape_vec + v)
        diff = self.shape_vec + v
        diff2 = self.shape_vec2 - v2
        #return np.mean(np.abs(diff)) / np.std(diff)
        #return np.sum(np.pow(diff, 2))
        shape_diff = np.std(diff)
        shape_diff2 = np.std(diff2)
        return_value = color_diff + shape_diff * 20 + shape_diff2 * 20

        self.CACHE[id] = return_value
        return return_value

    def get_prev(self) -> Edge:
        return self.piece.edges[(self.idx - 1) % 4]

    def get_next(self) -> Edge:
        return self.piece.edges[(self.idx + 1) % 4]

    def __repr__(self):
        return f"<Edge {self.idx} in piece {self.piece.id} - {self.type.name}>"

from __future__ import annotations
from enum import IntEnum
from typing import TYPE_CHECKING
import cv2
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
            up = np.abs(s)
            vect.append(up / down)

        return np.array(vect) * self.type.value

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



            
            
from __future__ import annotations
from enum import IntEnum

import cv2
import numpy as np
from edge import Edge, EdgeType
from utils import barycenter



class PieceType(IntEnum):
    CORNER=0
    EDGE=1
    INSIDE=2


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
            edge_types: list[EdgeType],
            edge_contours: list[np.ndarray],
            img:np.ndarray):
        self.id: int = id
        self.contour: np.array = contour
        self.corners_pos: np.array = corners
        self.edges: list[Edge] = [
            Edge(self, i, cont, typ,img)
            for i, (cont, typ) in enumerate(zip(edge_contours, edge_types))
        ]
        self.bbox: tuple[int, int, int, int] = bbox
        self.img: np.array = img
        self.center: np.array = tuple(barycenter(self.contour))

        flat_length = edge_types.count(EdgeType.FLAT)
        if flat_length == 2:
            self.type = PieceType.CORNER
        elif flat_length == 1:
            self.type = PieceType.EDGE
        else:
            self.type = PieceType.INSIDE

    def get_edge_by_offset(self, top_index: int, dx: int, dy: int) -> Edge:
        index = top_index
        if dx == 1:
            index += 3
        elif dx == -1:
            index += 1
        elif dy == 1:
            index += 2
        index %= 4
        return self.edges[index]

    def draw(self, img: np.ndarray):
        for edge in self.edges:
            col = self.EDGE_COLORS[edge.type]
            cv2.polylines(img, [edge.contour], False, col, 2)

        for pt in np.int32(self.corners_pos):
            cv2.circle(img, pt, 5, self.CORNER_COLOR, -1)

        cv2.putText(
            img,
            str(self.id),
            self.center,
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (255, 255, 255),
            3
        )

    def get_base_angle(self) -> float:
        center = np.mean(self.corners_pos, axis=0)
        angle = np.atan2(
            self.corners_pos[0][1] - center[1],
            self.corners_pos[0][0] - center[0]
        )
        return angle

    def get_rotated_img(self, rot: int) -> np.array:
        base_angle: float = self.get_base_angle() * 180 / np.pi
        masked = cv2.cvtColor(self.img, cv2.COLOR_BGR2BGRA)
        mask = np.zeros(masked.shape[:2], np.uint8)
        cv2.drawContours(mask, [self.contour], -1, (255, 255, 255, 255), cv2.FILLED)
        cv2.circle(masked, np.intp(self.corners_pos[0]), 16, (255, 0, 255, 255), cv2.FILLED)
        cv2.circle(masked, self.center, 2, (0, 0, 255, 255), cv2.FILLED)
        #cv2.imwrite("__1mask.png", mask)
        #cv2.imshow("mask", mask)
        masked[..., 3] = mask

        w = masked.shape[1]
        h = masked.shape[0]
        mat = cv2.getRotationMatrix2D(np.float32(self.center), base_angle + 45 - rot * 90, 1)
        abs_cos = abs(mat[0,0])
        abs_sin = abs(mat[0,1])
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)
        mat[0, 2] += bound_w/2 - w/2
        mat[1, 2] += bound_h/2 - h/2
        rotated = cv2.warpAffine(masked, mat, (bound_w, bound_h))
        rotated_mask = cv2.warpAffine(mask, mat, (bound_w, bound_h))
        #cv2.imwrite("__3rotimg.png", rotated)
        #cv2.imshow("rotated", rotated)
        #cv2.imwrite("__4rotmask.png", rotated_mask)
        #cv2.imshow("rotated mask", rotated_mask)

        coords = cv2.findNonZero(rotated_mask)
        mx, my, mw, mh = cv2.boundingRect(coords)

        cropped = rotated[my:my+mh, mx:mx+mw]
        #cv2.imwrite("__5cropped.png", cropped)
        #cv2.imshow("cropped", cropped)

        #cv2.waitKey(-1)

        return cropped

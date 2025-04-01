from __future__ import annotations

import cv2
import numpy as np

from edge import Edge, EdgeType
from utils import barycenter


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
            edge_contours: list[np.ndarray],
            img:np.ndarray):
        self.id: int = id
        self.contour: np.array = contour
        self.corners_pos: np.array = corners
        self.edges: list[Edge] = [
            Edge(self, i, cont, typ,img)
            for i, (cont, typ) in enumerate(zip(edge_contours, edges))
        ]
        self.bbox: tuple[int, int, int, int] = bbox
        self.img: np.array = img
        self.center: np.array = tuple(barycenter(self.contour))

    def draw(self, img: np.ndarray):
        for edge in self.edges:
            col = self.EDGE_COLORS[edge.type]
            cv2.polylines(img, [edge.contour], False, col, 2)

        for pt in np.int32(self.corners_pos):
            cv2.circle(img, pt, 3, self.CORNER_COLOR, -1)

    def get_base_angle(self) -> float:
        center = np.mean(self.corners_pos, axis=0)
        angle = np.atan2(
            self.corners_pos[0][1] - center[1],
            self.corners_pos[0][0] - center[0]
        )
        return angle

    def get_rotated_img(self, rot: int) -> np.array:
        base_angle: float = self.get_base_angle() * 180 / np.pi
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2BGRA)
        mat = cv2.getRotationMatrix2D(np.float32(self.center), base_angle + 45 - rot * 90, 1)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [self.contour], -1, (255, 255, 255, 255), cv2.FILLED)
        cv2.circle(img, np.intp(self.corners_pos[0]), 16, (255, 0, 255, 255), cv2.FILLED)
        cv2.circle(img, self.center, 2, (0, 0, 255, 255), cv2.FILLED)
        cv2.imwrite("__1mask.png", mask)
        rotated_mask = cv2.warpAffine(mask, mat, img.shape[:2])
        cv2.imwrite("__2rotmask.png", rotated_mask)
        rotated_img = cv2.warpAffine(img, mat, img.shape[:2])
        cv2.imwrite("__3rotimg.png", rotated_img)
        #out = np.zeros([img.shape[0], img.shape[1], 4], np.uint8)
        #cv2.copyTo(rotated_img, rotated_mask, out)
        out = rotated_img.copy()
        out[..., 3] = rotated_mask
        cv2.imwrite("__4out.png", out)
        coords = cv2.findNonZero(rotated_mask)
        x, y, w, h = cv2.boundingRect(coords)
        #cv2.waitKey(-1)
        cv2.imwrite("__5cropped.png", out[y:y+h, x:x+w, :])
        return out[y:y+h, x:x+w, :]

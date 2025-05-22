from __future__ import annotations

from enum import IntEnum

import cv2
import imutils
import numpy as np

from edge import Edge, EdgeType
from utils import barycenter, to_polar, pol2cart


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

    def extract_piece_img(self):
        x, y, w, h = cv2.boundingRect(self.contour)
        cropped = self.img[y:y+h, x:x+w]
        mask = np.zeros((h, w), dtype=np.uint8)
        contour = self.contour - np.array([x, y])
        cv2.drawContours(mask, [contour], 0, (255, 255, 255), cv2.FILLED)

        self.masked = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        self.masked[..., 3] = mask

    def warp_place(self, corners_pos: list[np.array], target_corners_pos: list[tuple[float, float]], img: np.array):
        x, y, w, h = cv2.boundingRect(self.contour)
        corners_pos = np.float32(corners_pos)
        corners_pos[..., 0] -= x
        corners_pos[..., 1] -= y

        # Crop image around piece and draw alpha mask
        M = cv2.getPerspectiveTransform(corners_pos, np.float32(target_corners_pos))
        warped = cv2.warpPerspective(self.masked, M, img.shape[:2][::-1])
        #print(img.shape, warped.shape)
        #print(corners_pos, target_corners_pos)

        mask = warped[..., 3] != 0
        img[mask] = warped[mask]

    def get_rotated_img(self, top_idx: int, vec: np.array, is_side: bool = False) -> np.array:
        # Crop image around piece and draw alpha mask
        x, y, w, h = cv2.boundingRect(self.contour)
        cropped = self.img[y:y+h, x:x+w]
        mask = np.zeros((h, w), dtype=np.uint8)
        contour = self.contour - np.array([x, y])
        cv2.drawContours(mask, [contour], 0, (255, 255, 255), cv2.FILLED)
        #cv2.imwrite(f"/tmp/imgs/piece_{self.id:03d}.jpg", cropped)
        #cv2.imwrite(f"/tmp/imgs/mask_{self.id:03d}.jpg", mask)

        masked = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        masked[..., 3] = mask

        # Take top edge, or left edge is `is_side` is `True`
        i: int = (top_idx + 1) % 4 if is_side else top_idx

        # Get corners in order (tl, bl, br, tr), relative to cropped image top-left corner
        corners = np.array([
            np.array(self.corners_pos[(j + top_idx + 1) % 4]) - (x, y)
            for j in range(4)
        ])

        center: np.array = np.array(mask.shape[::-1], dtype=np.float32) / 2
        #cv2.circle(masked, np.intp(center), 5, (0, 0, 255, 255), cv2.FILLED)
        #cv2.circle(masked, np.intp(corners[0]), 5, (120, 120, 255, 255), cv2.FILLED)
        #cv2.circle(masked, np.intp(corners[1]), 5, (120, 255, 120, 255), cv2.FILLED)
        #cv2.circle(masked, np.intp(corners[2]), 5, (255, 120, 120, 255), cv2.FILLED)
        #cv2.circle(masked, np.intp(corners[3]), 5, (120, 255, 255, 255), cv2.FILLED)

        #cv2.imwrite(f"/tmp/imgs/masked_{self.id:03d}.png", masked)

        c1: np.array = np.array(self.corners_pos[i])
        c2: np.array = np.array(self.corners_pos[(i + 1) % 4])
        edge_vec: np.array = c2 - c1

        # Compute angle and scale to match edge_vec and vec
        norm_vec: float = np.linalg.norm(vec)
        norm_edge_vec: float = np.linalg.norm(edge_vec)
        angle: float = np.atan2(edge_vec[0]*vec[1] - edge_vec[1]*vec[0], edge_vec[0]*vec[0] + edge_vec[1]*vec[1])
        angle_deg: float = angle * 180 / np.pi
        scale: float = norm_vec / norm_edge_vec

        # Rotate and scale image
        rotated = imutils.rotate_bound(masked, angle_deg)
        rotated = cv2.resize(rotated, (int(rotated.shape[1] * scale), int(rotated.shape[0] * scale)))
        #cv2.imwrite(f"/tmp/imgs/rotated_{self.id:03d}.png", rotated)

        # Rotate and scale corners
        new_center = (rotated.shape[1] / 2, rotated.shape[0] / 2)
        polar = to_polar(corners, center=center)

        angles = polar[..., 0]
        dists = polar[..., 1]

        angles += angle
        dists *= scale

        new_x, new_y = pol2cart([angles, dists])
        rotscaled = np.array(list(zip(new_x, new_y)))
        new_corners = np.intp(rotscaled + new_center)

        #with_corners = np.copy(rotated)
        #cv2.circle(with_corners, new_corners[0], 5, (0, 0, 255, 255), cv2.FILLED)
        #cv2.circle(with_corners, new_corners[1], 5, (0, 255, 0, 255), cv2.FILLED)
        #cv2.circle(with_corners, new_corners[2], 5, (255, 0, 0, 255), cv2.FILLED)
        #cv2.circle(with_corners, new_corners[3], 5, (0, 255, 255, 255), cv2.FILLED)
        #cv2.imwrite(f"/tmp/imgs/corners_{self.id:03d}.png", with_corners)

        return rotated, new_corners

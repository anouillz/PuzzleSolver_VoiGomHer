from __future__ import annotations

import random
import time
from typing import Optional

import cv2
import numpy as np

from contours import find_piece_contours
from corners import find_corners
from edge import Edge, EdgeType
from piece import Piece, PieceType


OFFSETS = [
    ( 0, -1),
    (-1,  0),
    ( 0,  1),
    ( 1,  0),
]


class Puzzle:
    def __init__(self, pieces: list[Piece], img: Optional[np.array] = None):
        self.img: Optional[np.array] = img
        self.pieces: list[Piece] = pieces
        self.solution: Optional[np.array] = None
        self.sol_i = 0
        self.max_depth = 0

    def __repr__(self):
        return f"<Puzzle {len(self.pieces)} pieces - {'UNSOLVED' if self.solution is None else 'SOLVED'}>"

    @staticmethod
    def from_image(img: np.array, img2: np.array, img3: np.array, img4: np.array) -> Puzzle:
        # Contours
        print("Finding contours")
        piece_contours = find_piece_contours(img, img2, img3, img4)
        n_pieces = len(piece_contours)
        print(f"{n_pieces} pieces")

        # Corners + edges
        print("Identifying corners and edges")
        pieces = []
        img2 = img.copy()
        for i, (contour, box) in enumerate(piece_contours):
            corners, edges, edge_contours = find_corners(contour)
            piece = Piece(i, contour, box, corners, edges, edge_contours, img)
            pieces.append(piece)
            piece.draw(img2)

        cv2.imwrite(f"/tmp/img.jpg", img2)
        return Puzzle(pieces, img)

    @staticmethod
    def from_images(imgs: list[tuple[np.array, np.array, np.array, np.array]]) -> Puzzle:
        pieces = []
        piece_id = 0
        for i, wrgb_imgs in enumerate(imgs):
            print(f"Processing image {i+1}/{len(imgs)}")
            img = wrgb_imgs[0]

            # Contours
            print("Finding contours")
            piece_contours = find_piece_contours(*wrgb_imgs)
            n_pieces = len(piece_contours)
            print(f"{n_pieces} pieces")
            img2 = img.copy()

            # Corners + edges
            print("Identifying corners and edges")
            for contour, box in piece_contours:
                corners, edges, edge_contours = find_corners(contour)
                piece = Piece(piece_id, contour, box, corners, edges, edge_contours, img)
                pieces.append(piece)
                piece.draw(img2)
                piece_id += 1

            cv2.imwrite(f"/tmp/img_{i}.jpg", img2)
            #cv2.waitKey(-1)

        return Puzzle(pieces, None)

    def show_solution(self, show: bool = True, save: bool = False):
        if self.solution is None:
            raise ValueError("Cannot show solution because puzzle is not solved. Try to call the solve() method first")

        coords = np.argwhere(self.solution[..., 0] != -1)
        y, x, h, w = cv2.boundingRect(coords)
        #np.savetxt("/tmp/sol.txt", self.solution[..., 0], fmt="%4d")
        self.solution = self.solution[y:y+h,x:x+w]
        print(self.solution.shape)
        print(self.solution[..., 0])

        corners_pos: np.array = np.zeros((self.solution.shape[0] + 1, self.solution.shape[1] + 1, 2), dtype=np.float32)

        corners_pos[0, 0] = [200, 200]
        corners_pos[0, 1] = [400, 200]

        img: np.array = np.zeros((0, 0, 4), np.uint8)
        for y in range(self.solution.shape[0]):
            for x in range(self.solution.shape[1]):
                idx, top_idx = self.solution[y, x]
                if idx == -1:
                    continue
                piece: Piece = self.pieces[idx]
                if x == 0:
                    c1 = corners_pos[y, x+1]  # top-right
                    c2 = corners_pos[y, x]  # top-left
                    ox, oy = c2
                    is_side = False
                else:
                    c1 = corners_pos[y, x]  # top-left
                    c2 = corners_pos[y+1, x]  # bottom-left
                    ox, oy = c1
                    is_side = True

                vec: np.array = c2 - c1
                res: tuple[np.array, tuple[np.array, np.array, np.array, np.array]] = piece.get_rotated_img(top_idx, vec, is_side)
                piece_img: np.array = res[0]
                tl, bl, br, tr = res[1]

                if is_side:
                    corners_pos[y, x+1] = np.array([ox + tr[0] - tl[0], oy + tr[1] - tl[1]])  # top-right
                    corners_pos[y+1, x+1] = np.array([ox + br[0] - tl[0], oy + br[1] - tl[1]])  # bottom-right
                else:
                    corners_pos[y+1, x] = np.array([ox + bl[0] - tl[0], oy + bl[1] - tl[1]])  # bottom-left
                    corners_pos[y+1, x+1] = np.array([ox + br[0] - tl[0], oy + br[1] - tl[1]])  # bottom-right

                x1 = int(ox - tl[0])
                y1 = int(oy - tl[1])
                x2 = x1 + piece_img.shape[1]
                y2 = y1 + piece_img.shape[0]

                mask = piece_img[..., 3] != 0
                if x2 >= img.shape[1] or y2 >= img.shape[0]:
                    #print(f"Increasing image size to include ({x1}:{x2}, {y1}:{y2})  (current size: {img.shape[1]}x{img.shape[0]})")
                    img2 = np.zeros((max(img.shape[0], y2), max(img.shape[1], x2), 4), np.uint8)
                    img2[:img.shape[0], :img.shape[1]] = img
                    img = img2

                if y1 < 0 or x1 < 0:
                    print(f"Oops, we're going out of the image ({x},{y}), ({x1}, {y1})")
                img[y1:y2, x1:x2][mask] = piece_img[mask]

                #img2 = np.copy(img)
                #for cy in range(corners_pos.shape[0]):
                #    for cx in range(corners_pos.shape[1]):
                #        cv2.circle(img2, np.intp(corners_pos[cy, cx]), 5, (255, 255, 255), cv2.FILLED)
                #cv2.imshow("Solution", img2)
                #cv2.waitKey(-1)

        if show:
            cv2.imshow("Solution", img)
            cv2.waitKey(100)
        if save:
            cv2.imwrite(f"/tmp/solution_{self.sol_i}.jpg", img)
            self.sol_i += 1
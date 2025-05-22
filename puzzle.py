from __future__ import annotations

import random
import time
from typing import Optional

import cv2
import numpy as np

from contours import find_piece_contours
from corners import find_corners
from edge import Edge, EdgeType
from mesh import Mesh
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

        mesh: Mesh = Mesh.from_puzzle(self)
        steps = mesh.stabilize(10000)
        print(f"Stabilized in {steps} steps")

        ox, oy, w, h = mesh.get_bounds()

        margin: float = 20
        img_size: tuple[int, int, int] = (
            int(h + margin * 2),
            int(w + margin * 2),
            4
        )

        img: np.array = np.zeros(img_size, np.uint8)
        offsets = [(0, 0), (0, 1), (1, 1), (1, 0)]

        for piece in self.pieces:
            piece.extract_piece_img()

        t0 = time.time()
        for y in range(self.solution.shape[0]):
            for x in range(self.solution.shape[1]):
                idx, top_idx = self.solution[y, x]
                if idx == -1:
                    continue
                piece: Piece = self.pieces[idx]

                corners_pos = []
                target_corner_pos = []
                for i, (dx, dy) in enumerate(offsets):
                    node = mesh.node_grid[y * 2 + dy][x * 2 + dx]
                    target_corner_pos.append((node.pos.x - ox + margin, node.pos.y - oy + margin))
                    corners_pos.append(piece.corners_pos[(i + top_idx + 1) % 4])

                piece.warp_place(corners_pos, target_corner_pos, img)

        t1 = time.time()
        print(f"Created solution image in {t1-t0:.2f}s")
        if show:
            cv2.imshow("Solution", img)
            cv2.waitKey(100)
        if save:
            cv2.imwrite(f"/tmp/solution_{self.sol_i}.jpg", img)
            self.sol_i += 1
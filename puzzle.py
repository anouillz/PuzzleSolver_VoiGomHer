from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from contours import find_piece_contours
from corners import find_corners
from piece import Piece


class Puzzle:
    def __init__(self, pieces: list[Piece], img: Optional[np.array] = None):
        self.img: Optional[np.array] = img
        self.pieces: list[Piece] = pieces
        self.solution: Optional[np.array] = None

    def __repr__(self):
        return f"<Puzzle {len(self.pieces)} pieces - {"UNSOLVED" if self.solution is None else "SOLVED"}>"

    @staticmethod
    def from_image(img: np.array) -> Puzzle:
        # Contours
        print("Finding contours")
        piece_contours = find_piece_contours(img)
        n_pieces = len(piece_contours)
        print(f"{n_pieces} pieces")

        # Corners + edges
        print("Identifying corners and edges")
        pieces = []
        for i, (contour, box) in enumerate(piece_contours):
            corners, edges, edge_contours = find_corners(contour)
            piece = Piece(i, contour, box, corners, edges, edge_contours, img)
            pieces.append(piece)

        return Puzzle(pieces, img)

    @staticmethod
    def from_images(imgs: list[np.array]) -> Puzzle:
        pieces = []
        piece_id = 0
        for i, img in enumerate(imgs):
            print(f"Processing image {i+1}/{len(imgs)}")

            # Contours
            print("Finding contours")
            piece_contours = find_piece_contours(img)
            n_pieces = len(piece_contours)
            print(f"{n_pieces} pieces")

            # Corners + edges
            print("Identifying corners and edges")
            for contour, box in piece_contours:
                corners, edges, edge_contours = find_corners(contour)
                piece = Piece(piece_id, contour, box, corners, edges, edge_contours, img)
                pieces.append(piece)
                piece_id += 1

        return Puzzle(pieces, None)

    def solve(self):
        pass

    def show_solution(self):
        if self.solution is None:
            raise ValueError("Cannot show solution because puzzle is not solved. Try to call the solve() method first")

        img: np.array = np.zeros((0, 0, 4), np.uint8)
        oy = 0
        max_w = 0
        for y in range(self.solution.shape[0]):
            height = 0
            ox = 0
            for x in range(self.solution.shape[1]):
                idx, rot = self.solution[y, x]
                if idx == -1:
                    ox += max_w
                    continue
                piece: Piece = self.pieces[idx]
                piece_img: np.array = piece.get_rotated_img(rot)
                mask = piece_img[..., 3] != 0
                x2 = ox + piece_img.shape[1]
                y2 = oy + piece_img.shape[0]
                if x2 >= img.shape[1] or y2 >= img.shape[0]:
                    img2 = np.zeros((max(img.shape[0], y2), max(img.shape[1], x2), 4), np.uint8)
                    img2[:img.shape[0], :img.shape[1]] = img
                    img = img2
                img[oy:y2, ox:x2][mask] = piece_img[mask]
                height = max(height, piece_img.shape[0])
                ox += piece_img.shape[1]
                max_w = max(max_w, piece_img.shape[1])
                #cv2.imshow("Solution", img)
                #cv2.waitKey(100)
            oy += height

        cv2.imshow("Solution", img)
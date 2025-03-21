import time

import cv2
import numpy as np
from itertools import combinations
import tqdm
from matplotlib import pyplot as plt

from edge import EdgeType
from piece import Piece
from utils import resample_polygon

timings = {
    "resampling": [],
    "polar": [],
    "peaks": [],
    "kurtosis": [],
    "corners": [],
    "edges": [],
    "total": [],
}

# ChatGPT
def fast_kurtosis(arr):
    """Compute kurtosis using NumPy, optimized for speed."""
    n = len(arr)
    mean = np.mean(arr)
    diff = arr - mean
    var = np.mean(diff**2)  # Variance
    kurt = np.mean(diff**4) / (var**2)  # Kurtosis formula
    return kurt - 3  # Excess kurtosis (same as SciPy)

def angle_diff(angle1, angle2):
    diff = np.abs(angle1 - angle2) % (np.pi * 2)
    if diff > np.pi:
        diff = np.pi * 2 - diff
    return diff

def to_polar(contour):
    cv_contour = contour.reshape((contour.shape[0], 1, contour.shape[1]))
    cv_contour = np.int32(cv_contour)
    M = cv2.moments(cv_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    distances = []
    angles = []

    for c in contour:
        dist = np.linalg.norm(c - [cX, cY])
        angle = np.atan2(c[1] - cY, c[0] - cX)
        distances.append(dist)
        angles.append(angle)

    return np.array(list(zip(angles, distances)))

def wrap_win(data, center, radius):
    return np.array(data).take(
        range(
            int(center) - radius,
            int(center) + radius + 1
        ),
        mode="wrap"
    )

def wrap_win_func(data, radius, func):
    res = []
    for i in range(len(data)):
        res.append(func(wrap_win(data, i, radius)))
    return res

def wrap_mean(data, radius):
    return wrap_win_func(data, radius, np.mean)

def filter_corners(angles, dists, candidate_pairs):
    candidates = np.uint32(np.array(candidate_pairs)[:, 0])

    # Try all groups of 4 corners
    for comb in combinations(candidates, 4):
        comb = list(comb)
        comb_dists = dists[comb]
        comb_angles = angles[comb]

        cartesian = np.stack([comb_dists, comb_dists], axis=1) * np.stack([
            np.cos(comb_angles),
            np.sin(comb_angles),
        ], axis=1)
        center = np.mean(cartesian, axis=0)
        new_dists = []
        new_angles = []

        for pt in cartesian:
            dist = np.linalg.norm(pt - center)
            angle = np.atan2(pt[1] - center[1], pt[0] - center[0])
            new_dists.append(dist)
            new_angles.append(angle)

        angle1 = angle_diff(new_angles[0], new_angles[1])
        angle2 = angle_diff(new_angles[2], new_angles[3])

        angle3 = angle_diff(new_angles[0], new_angles[3])
        angle4 = angle_diff(new_angles[1], new_angles[2])

        diff1 = angle_diff(angle1, angle2)
        diff2 = angle_diff(angle3, angle4)

        if np.std(new_dists) / np.mean(new_dists) < 0.1 and diff1 < 0.3 and diff2 < 0.3:
            return comb

    print("No perfect match")
    return candidates[:4]

def pol2cart(pol):
    angle, dist = pol
    return [
        dist * np.cos(angle),
        dist * np.sin(angle)
    ]

def find_corners(contour, dbg=False):
    if dbg: print("Resampling")
    t0 = time.time()
    contour = resample_polygon(contour[:, 0, :], 360)
    og_contour = contour
    t1 = time.time()
    timings["resampling"].append(t1 - t0)

    if dbg: print("Converting to polar coordinates")
    t0 = time.time()
    contour = to_polar(contour)

    angles = contour[:, 0]
    dists = contour[:, 1]

    N = contour.shape[0]
    t1 = time.time()
    timings["polar"].append(t1 - t0)

    if dbg: print("Finding peaks")
    t0 = time.time()
    peaks = []
    for i in range(N):
        win = wrap_win(dists, i, 4)
        if dists[i] == np.max(win):
            peaks.append(i)

    t1 = time.time()
    timings["peaks"].append(t1 - t0)

    if dbg: print("Computing kurtosis")
    t0 = time.time()
    #kurt = wrap_win_func(dists, 4, lambda win: scipy.stats.kurtosis(win))
    kurt = wrap_win_func(dists, 8, fast_kurtosis)
    t1 = time.time()
    timings["kurtosis"].append(t1 - t0)

    if dbg: print("Identifying corners")
    t0 = time.time()
    pairs = [
        (peak, np.mean(wrap_win(kurt, peak, 8)))
        for peak in peaks
    ]
    pairs = sorted(pairs, key=lambda p: p[1])
    corners = sorted(filter_corners(angles, dists, pairs))
    t1 = time.time()
    timings["corners"].append(t1 - t0)

    if dbg:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax2 = ax.twinx()
        ax.scatter(range(N), dists, s=5, label="Distances")
        ax2.plot(range(N), wrap_mean(kurt, 8), c="r", label="Smoothed kurtosis")
        ax.vlines(corners, np.min(dists), np.max(dists), label="Selected corners")

    tmp = np.zeros((3, 4))
    t0 = time.time()
    edges = []
    edge_contours = []
    for i in range(4):
        c1, c2 = corners[i], corners[(i + 1) % 4]
        c1cart = pol2cart(contour[c1])
        c2cart = pol2cart(contour[c2])
        if i == 3:
            c2 += N
        cm = (c1 + c2) // 2 % N
        mid = [
            (c1cart[0] + c2cart[0]) / 2,
            (c1cart[1] + c2cart[1]) / 2
        ]
        mid_dist = np.linalg.norm(mid)
        mean_dist = np.mean(wrap_win(dists, cm, 4))
        tmp[0, i] = cm
        tmp[1, i] = mid_dist
        tmp[2, i] = mean_dist
        ratio = mean_dist / mid_dist

        edge = EdgeType.FLAT
        if ratio < 0.9:
            edge = EdgeType.INSIDE
        elif ratio > 1.1:
            edge = EdgeType.OUTSIDE
        edges.append(edge)
        cont = og_contour.take(
            range(c1, c2 + 1),
            axis=0,
            mode="wrap"
        )
        cont = np.int32(cont.reshape((cont.shape[0], 1, cont.shape[1])))
        edge_contours.append(cont)

    t1 = time.time()
    timings["edges"].append(t1 - t0)

    if dbg:
        ax.scatter(tmp[0], tmp[1], label="Edge distance")
        ax.scatter(tmp[0], tmp[2], label="Mean contour distance")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(
            lines + lines2,
            labels + labels2,
            bbox_to_anchor=(1, 1),
            loc="upper left"
        )
        plt.tight_layout()
        plt.show()

    pts = og_contour[corners]
    return pts, edges, edge_contours

def edge_length(corners):
    lengths = []
    # compute length with its 2 corners and its length in a 3 element list
    for i in range(4):
        c1, c2 = corners[i], corners[(i + 1) % 4]
        length = np.linalg.norm(c1 - c2)
        lengths.append(length)

    return lengths


def edge_vect(edge):
    vect = []
    c1 = edge[0]
    c2 = edge[-1]
    line = c2 - c1
    n = np.array([-line[1], line[0]])
    down = np.linalg.norm(n)

    e = resample_polygon(edge[:, 0, :], 30)

    for i in e:
        v = i - c1
        s = np.dot(v, n)
        up = np.abs(s)
        vect.append(up / down)

    return np.array(vect)

def main():
    from contours import find_piece_contours

    #img = cv2.imread("res/49_pieces_sweater_kevin_small.jpg")
    #img = cv2.imread("res/49_pieces_sweater_kevin_small2.jpg")
    #img = cv2.imread("res/24_pieces_sweater_kevin.jpg")
    img = cv2.imread("res/24.jpg")
    #img = cv2.imread("res/24_pieces3_black_3.jpg")
    pieces = find_piece_contours(img)
    n_pieces = len(pieces)
    print(f"{n_pieces} pieces")

    dbg = False
    piece_by_piece = False

    for i, (contour, box) in tqdm.tqdm(enumerate(pieces)):
        #cv2.drawContours(img, [contour], 0, (0, 0, 255), 2)
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw number in center of piece
        cv2.putText(img, str(i), (cX, cY), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 120), 3)

        x0 = np.min(box[:, 0])
        y0 = np.min(box[:, 1])
        x1 = np.max(box[:, 0])
        y1 = np.max(box[:, 1])

        t0 = time.time()
        corners, edges, edge_contours = find_corners(contour, dbg)
        t1 = time.time()
        timings["total"].append(t1 - t0)
        p = Piece(i, contour, box, corners, edges, edge_contours)
        p.draw(img)

        """
        corners = np.array(corners)
        cont = np.int32(corners.reshape((corners.shape[0], 1, corners.shape[1])))
        cv2.drawContours(img, [cont], 0, (0, 255, 0), 2)
        for j in range(4):
            c1, c2 = corners[j], corners[(j + 1) % 4]
            cm = [(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2]
            col = (0, 255, 255)
            edge = edges[j]
            if edge == OUTSIDE:
                col = (0, 255, 0)
            elif edge == INSIDE:
                col = (0, 0, 255)
            cv2.drawContours(img, [edge_contours[j]], -1, col, 2)
            cv2.circle(img, (int(cm[0]), int(cm[1])), 5, col, -1)
        """

        for i in range(4):
            edge = p.edges[i]
            plt.plot(range(edge.SHAPE_VEC_SIZE), edge.shape_vec, label=f"Edge {i}")
        plt.legend()
        plt.show()

        if piece_by_piece and y0 > 0 and x0 > 0 and y1 - y0 > 0 and x1 - x0 > 0:
            cv2.imshow("piece", img[y0:y1, x0:x1])
            cv2.setWindowTitle("piece", f"Piece {i+1}/{n_pieces}")
            cv2.waitKey(-1)

    cv2.imshow("Pieces", img)
    cv2.waitKey(-1)

    for k, v in timings.items():
        print(f"{k.title()}: {sum(v) / len(v)}")


if __name__ == '__main__':
    main()
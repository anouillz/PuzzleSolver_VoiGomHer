import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy

from contours import find_piece_contours


def wrap_win(data, center, radius):
    res = []
    for i in range(center - radius, center + radius + 1):
        res.append(data[i % len(data)])
    return res

def wrap_win_func(data, radius, func):
    res = []
    for i in range(len(data)):
        res.append(func(wrap_win(data, i, radius)))
    return res

def wrap_mean(data, radius):
    return wrap_win_func(data, radius, np.mean)

def test():
    fig, axes = plt.subplots(figsize=(10, 10), nrows=3)
    contour = np.loadtxt("polar_contour.csv", delimiter=",")
    #axes[0].plot(contour[:, 0], contour[:, 1])
    #axes[0].scatter(contour[:, 0], contour[:, 1], s=5)

    smoothed = wrap_mean(contour[:, 1], 3)
    #axes[0].scatter(range(contour.shape[0]), contour[:, 1], s=5)
    axes[0].scatter(range(contour.shape[0]), smoothed, s=5)

    #ax2 = ax1.twinx()
    #ax2.scatter(range(contour.shape[0]), contour[:, 0], s=5, color="red")

    """
    for ax in axes:
        ax.hlines(0, 0, contour.shape[0], color="green")
    """
    dist = contour.shape[0] // 12
    axes[0].vlines(
        #scipy.signal.argrelmax(contour[:, 1])[0],
        scipy.signal.find_peaks(
            contour[:, 1],
            distance=dist,

        )[0],
        np.min(contour[:, 1]), np.max(contour[:, 1])
    )

    deriv = []
    for i in range(contour.shape[0]):
        a2, d2 = contour[i]
        a1, d1 = contour[(i - 1) % contour.shape[0]]
        d = (d2 - d1) / ((a2 - a1) % (2 * np.pi))
        #d = min(max(d, -2), 2)
        #d = d2 - d1
        deriv.append(d)
    deriv = np.array(deriv)

    colors = []
    outlier = np.abs(deriv) > 100
    changes = []

    kurt = []
    for i in range(contour.shape[0]):
        if abs(deriv[i]) < 100:
            colors.append("green")
        else:
            colors.append("red")

        #if outlier[i] and not der
        pts = []
        n = 10
        for j in range(i - n, i + n + 1):
            pts.append(contour[j % contour.shape[0], 1])

        k = scipy.stats.kurtosis(np.array(pts))
        kurt.append(k)

    kurt2 = wrap_mean(kurt, 5)

    #deriv2 = wrap_mean(deriv, 4)

    """
    deriv2 = []
    for i in range(contour.shape[0]):
        d2 = deriv[i]
        d1 = deriv[(i - 1) % contour.shape[0]]
        deriv2.append(d2 - d1)
    """

    #ax2.scatter(contour[:, 0], deriv, color="red", s=3)
    #axes[1].scatter(range(contour.shape[0]), deriv, color=colors, s=3)
    axes[1].scatter(range(contour.shape[0]), kurt, color="red", s=3)
    axes[2].scatter(range(contour.shape[0]), kurt2, color="red", s=3)

    a = axes[0].twinx()
    a.scatter(range(contour.shape[0]), kurt2, color="red", s=3)
    """
    peaks = scipy.signal.find_peaks(
        contour[:, 1],
        width=1,
        distance=contour.shape[0] / 10
    )[0]
    
    peaks_x = contour[:, 0][peaks]
    deriv = np.der
    
    print(peaks)
    print(peaks_x)
    plt.vlines(peaks_x, np.min(contour[:, 1]), np.max(contour[:, 1]))
    """
    plt.show()

def find_corners(contour, id, image, offset, debug=True):
    if debug:
        #fig, axes = plt.subplots(figsize=(10, 10), nrows=3)
        fig, axes = plt.subplots(figsize=(8, 6), nrows=2)
        ax = axes[0]
        ax2 = ax.twinx()

    #og_contour = contour[::2]
    #contour = to_polar(contour)[::2]
    og_contour = contour
    contour = to_polar(contour)

    #contour = np.loadtxt("polar_contour.csv", delimiter=",")
    angles = contour[:, 0]
    dists = contour[:, 1]

    dists = np.array(wrap_mean(dists, 2))
    N = contour.shape[0]
    indices = list(range(N))

    if debug:
        ax.scatter(indices, dists, s=5, label="Distances")
        d_angle = angles - np.concatenate([angles[1:], angles[:1]])
        ax2.scatter(indices, d_angle, s=5, color="green", label="Angles")

    deriv = []
    for i in range(N):
        d = dists[i]
        d2 = dists[(i + 1) % N]
        deriv.append(d2 - d)

    deriv2 = []
    for i in range(N):
        d = deriv[i]
        d2 = deriv[(i + 1) % N]
        deriv2.append(d2 - d)

    #ax2.scatter(indices, deriv, s=3, color="lime", label="1st derivative", marker="x")
    #ax2.scatter(indices, deriv2, s=3, color="red", label="2nd derivative", marker="x")

    # Find peaks
    min_peak_dist = N // 12
    t0 = time.time()
    """
    peaks = scipy.signal.find_peaks(
        dists,
        distance=min_peak_dist,
        #prominence=1,
        #wlen=21
    )[0]
    """
    peaks = []
    for i in range(N):
        win = wrap_win(dists, i, 7)
        if dists[i] == np.max(win):
            peaks.append(i)

    t1 = time.time()
    if debug:
        print(f"Found {len(peaks)} peaks in {t1 - t0:.3f}s")
        ax.vlines(
            peaks,
            np.min(dists), np.max(dists),
            label="Peaks"
        )

    t2 = time.time()
    kurt = wrap_win_func(dists, 6, lambda win: scipy.stats.kurtosis(win))
    kurt = wrap_mean(kurt, 2)

    t3 = time.time()
    if debug:
        print(f"Computed kurtosis in {t3 - t2:.3f}s")

        #ax2.scatter(indices, kurt, s=3, color="red", label="Kurtosis")

    corners = []
    t4 = time.time()
    p80 = np.percentile(kurt, 80)
    p25 = np.percentile(kurt, 25)
    p75 = np.percentile(kurt, 75)
    iqr = p75 - p25
    max_kurt = p75 + 1.5 * iqr
    mean_kurt = np.mean(kurt)
    std_kurt = np.std(kurt)
    max_kurt = p75

    mean_k = wrap_mean(kurt, 8)
    if debug:
        print(f"General mean={np.mean(kurt)}")
        ax2.hlines(max_kurt, 0, N)
        print(f"percentile 80%={p80}")
        ax2.scatter(indices, mean_k, s=3, color="red", label="Mean Kurtosis")
    tmp = []
    for i, peak in enumerate(peaks):
        k = kurt[peak]
        win = wrap_win(kurt, peak, 8)
        m = np.max(win)
        mean = np.mean(win)
        median = np.median(win)
        std = np.std(win)
        z_score = (mean - mean_kurt) / std_kurt
        ratio = mean / mean_kurt
        p75 = np.percentile(win, 75)
        if debug:
            print(f"Peak {i}: pos={peak}, kurtosis={k}, max={m}, std={std}, mean={mean}, median={median}")
            tmp.append(p75)
        if mean < max_kurt:
            if debug:
                print("  -> is a corner")
            corners.append(peak)

    #ax2.scatter(peaks, tmp, color="cyan", s=10)
    t5 = time.time()
    if debug:
        print(f"Filtered corners {len(peaks)} -> {len(corners)} in {t5 - t4:.3f}s")
        ax.scatter(corners, dists[corners], s=60, color="limegreen", label="Corners")
        axes[1].imshow(image)

    if debug:
        corner_pts = []
        for corner in corners:
            cx, cy = og_contour[corner][0]
            cx -= offset[0]
            cy -= offset[1]
            corner_pts.append((cx, cy))
        if len(corner_pts) != 0:
            corner_pts = np.array(corner_pts)
            axes[1].plot(corner_pts[:, 0], corner_pts[:, 1], marker="o")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="center left", bbox_to_anchor=(1.04, 0.5))
        plt.tight_layout(rect=[0, 0, 1, 1])

    #plt.close(fig)
    """if len(corners) != 4:
        if debug:
            plt.show()"""
    plt.show()
    #plt.savefig(f"/tmp/plot_{id}.png")

    print(f"Found {len(corners)} in {t5 - t4 + t3 - t2 + t1 - t0:.3f}s")
    return len(corners)

def to_polar(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    distances = []
    angles = []

    for c in contour:
        c = c[0]
        dist = np.linalg.norm(c - [cX, cY])
        angle = np.atan2(c[1] - cY, c[0] - cX)
        distances.append(dist)
        angles.append(angle)

    return np.array(list(zip(angles, distances)))

if __name__ == '__main__':
    img = cv2.imread("res/49_pieces_sweater_kevin_small.jpg")
    #img = cv2.imread("res/24_pieces_sweater_kevin.jpg")
    pieces = find_piece_contours(img)
    n_pieces = len(pieces)
    print(f"{n_pieces} pieces")
    correct = 0
    more = 0
    less = 0
    d = {}
    for i, (contour, box) in enumerate(pieces):
        c2 = np.array(contour[::2])

        cv2.drawContours(img, [contour], 0, (255, 0, 0), 1)
        cv2.drawContours(img, [c2], 0, (255, 255, 255), 1)
        x0 = np.min(box[:, 0])
        y0 = np.min(box[:, 1])
        x1 = np.max(box[:, 0])
        y1 = np.max(box[:, 1])
        print(f"Piece {i+1} / {n_pieces}")
        corners = find_corners(contour, i, img[y0:y1, x0:x1], (x0, y0))
        if corners not in d:
            d[corners] = 0
        d[corners] += 1
        c = input()
        if c == "export":
            np.savetxt("/tmp/data.csv", to_polar(contour))
        if corners != 4:
            if corners > 4:
                more += 1
            else:
                less += 1
            pass
        else:
            correct += 1

    print(f"Success rate: {correct} / {n_pieces} = {correct/n_pieces*100:.2f}%")
    print(f"More rate: {more} / {n_pieces} = {more/n_pieces*100:.2f}%")
    print(f"Less rate: {less} / {n_pieces} = {less/n_pieces*100:.2f}%")
    print(d)
    #cv2.imwrite("res/img_out/p24_1_pull_out_corners.jpg", img)

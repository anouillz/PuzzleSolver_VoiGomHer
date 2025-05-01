import cv2
import numpy as np

def barycenter(contour: np.array):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return np.array([cX, cY])

# https://stackoverflow.com/a/70664846/11109181
def resample_polygon(xy: np.ndarray, n_points: int = 100) -> np.ndarray:
    # Cumulative Euclidean distance between successive polygon points.
    # This will be the "x" for interpolation
    d = np.cumsum(np.r_[0, np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))])

    # get linearly spaced points along the cumulative Euclidean distance
    d_sampled = np.linspace(0, d.max(), n_points)

    # interpolate x and y coordinates
    xy_interp = np.c_[
        np.interp(d_sampled, d, xy[:, 0]),
        np.interp(d_sampled, d, xy[:, 1]),
    ]

    return xy_interp

def to_polar(contour, is_cv2_cnt: bool = False, center = None):
    pts = contour
    if is_cv2_cnt:
        pts = contour[:, 0, :].astype(np.float32)

    cX, cY = np.mean(pts, axis=0) if center is None else center

    distances = []
    angles = []

    for pt in pts:
        dist = np.linalg.norm(pt - [cX, cY])
        angle = np.arctan2(pt[1] - cY, pt[0] - cX)
        distances.append(dist)
        angles.append(angle)

    return np.array(list(zip(angles, distances)))

def pol2cart(pol):
    angle, dist = pol
    return [
        dist * np.cos(angle),
        dist * np.sin(angle)
    ]

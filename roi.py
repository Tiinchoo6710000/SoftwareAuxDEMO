import cv2
import numpy as np

def point_inside_roi(point, roi_polygon):
    return cv2.pointPolygonTest(roi_polygon, point, False) >= 0


def count_keypoints_inside_roi(keypoints, roi_polygon, min_conf=0.2):
    count = 0
    for kp in keypoints:
        x, y, conf = kp
        if conf < min_conf:
            continue
        if cv2.pointPolygonTest(roi_polygon, (int(x), int(y)), False) >= 0:
            count += 1
    return count


def bbox_roi_coverage(bbox, roi_polygon, grid=5):
    """
    Devuelve el porcentaje del bbox que cae dentro del ROI
    usando muestreo en grilla (robusto a ángulos raros)
    """
    x1, y1, x2, y2 = bbox
    inside = 0
    total = 0

    for gx in range(grid):
        for gy in range(grid):
            px = int(x1 + (gx + 0.5) * (x2 - x1) / grid)
            py = int(y1 + (gy + 0.5) * (y2 - y1) / grid)
            total += 1
            if cv2.pointPolygonTest(roi_polygon, (px, py), False) >= 0:
                inside += 1

    return inside / total if total > 0 else 0

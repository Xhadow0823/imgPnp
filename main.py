import cv2
import numpy as np
import os
from typing import List

CAMERA_MTX = np.array([
    [1.28637046e+03, 0.00000000e+00, 6.38763147e+02],
    [0.00000000e+00, 1.31271700e+03, 3.92384848e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.double)

DIST_COEF = np.array([
    [ 1.29466373e-01, -5.89043408e+00, -1.59987614e-02, -1.60014605e-03, 1.08631150e+02]
], dtype=np.double)

def draw_axis(img, R, t, C, D):
    rotV = R
    points = np.float32([[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, C, D)
    axisPoints = axisPoints.astype(np.int32)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

def draw_normal(img, root, R, T, C, D):
    line_pt1 = root  # objp[9]
    line_pt2 = line_pt1 + np.array([(0, 0, +40.0)], dtype=np.double).ravel()
    line_pts = np.array([line_pt1, line_pt2])
    line_pts, _ = cv2.projectPoints(line_pts, R, T, C, D)
    line_pt1 = line_pts[0].ravel().astype(np.int32)
    line_pt2 = line_pts[1].ravel().astype(np.int32)
    cv2.line(img, line_pt1, line_pt2, (255,255,0), 7)

class __Polygon:
    base_path: str = "./dataset"
    base_filename: str = None
    ratio_points: List[tuple[int, int]] = None
    camera_points: List = None
    is_exist: bool = False
    points_from_world = np.array([
            (-42.5,   5, 0),
            (42.5,   5, 0),
            (42.5,  -5, 0),
            (-42.5,  -5, 0)
        ], dtype="double"
    )
    rotation_vector = None
    translation_vector = None

    def __init__(self, name: str = None):
        self.base_filename = name

    def load(self, name: str = None):
        if name != None:
            self.base_filename = name
        filename = self.base_filename + ".txt"
        file_path = os.path.join(self.base_path, filename)
        with open(file_path, "rt") as f:
            line = f.read()
            raw_points = line.split(" ")[1:]
            self.ratio_points = np.array(list(map(lambda a: float(a), raw_points))).reshape((-1, 2))
        
        filename = self.base_filename + ".jpg"
        file_path = os.path.join(self.base_path, filename)
        if os.path.exists(file_path):
            self.is_exist = True
            img = cv2.imread(file_path)
            width, height, _ = img.shape
            self.camera_points = []
            for p in self.ratio_points:
                self.camera_points.append((int(p[0] * height), int(p[1] * width)))  # TODO ?????
            self.camera_points = np.array(self.camera_points, dtype=np.double)
            success, rotation_vector, translation_vector = cv2.solvePnP(self.points_from_world, self.camera_points, CAMERA_MTX, DIST_COEF, flags=0)
            if success:
                self.rotation_vector = rotation_vector
                self.translation_vector = translation_vector
    def show(self):
        filename = self.base_filename + ".jpg"
        file_path = os.path.join(self.base_path, filename)
        cv2.namedWindow("polygon")
        img = cv2.imread(file_path)
        cv2.imshow("polygon", img)
        return img
    
    def draw(self, img):
        # draw all points
        for p in self.camera_points:
            cv2.circle(img, p.astype(np.int32), 10, (0,255,255), -1)
        # draw normal line
        draw_normal(img, np.array([(0, 0, 0.0)], dtype=np.double), self.rotation_vector, self.translation_vector, CAMERA_MTX, DIST_COEF)
        
        # draw axis
        draw_axis(img, self.rotation_vector, self.translation_vector, CAMERA_MTX, DIST_COEF)
        
        # img = cv2.resize(img, (height//2, width//2))
        cv2.imshow("polygon", img)
        return img

if __name__ == "__main__":
    image_list = list(filter(lambda x: x.endswith('.txt'), next(os.walk("./dataset/"))[2]))
    image_list = list(map(lambda x: os.path.splitext(x)[0], image_list))
    
    for base_name in image_list:
        Polygon = __Polygon(base_name)
        Polygon.load()
        if not Polygon.is_exist:
            print(f"skip: {Polygon.base_filename}")
            continue
        img = Polygon.show()
        Polygon.draw(img)
        cv2.waitKey(-1)
    pass
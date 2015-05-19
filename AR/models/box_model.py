__author__ = 'ar1'


from model import Model
import numpy as np
import cv2


class BoxModel(Model):
    def __init__(self):
        Model.__init__(self)
        self._edges =  [(0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7)]

        self._sides = [(0, 1, 5, 4), (0, 4, 7, 3), (2, 3, 7, 6), (4, 5, 6, 7), (1, 2, 6, 5)]

        self._verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                                  [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])


    def draw(self,  frame, quad):
        """Draw on frame model of 3D box based on quad"""
        x0, y0, x1, y1 = (0, 0, frame.shape[0], frame.shape[1])
        quad_3d = np.float32([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])

        h, w = frame.shape[:2]
        camera_matrix = np.float64([[w,    0,    0.5*(w-1)],
                                    [0,    w,    0.5*(h-1)],
                                    [0,    0,    1        ]])

        dist_coef = np.zeros(4)
        ret, rvec, tvec = cv2.solvePnP(quad_3d, quad, camera_matrix, dist_coef)
        verts = self.verts * [(x1-x0), (y1-y0), -(x1-x0)*0.5] + (x0, y0, 0)
        verts = cv2.projectPoints(verts, rvec, tvec, camera_matrix, dist_coef)[0].reshape(-1, 2)

        for i, j, k, g in self.sides:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = verts[i], verts[j], verts[k], verts[g]
            side = (x0, y0), (x1, y1), (x2, y2), (x3, y3)
            # cv2.polylines(frame, [np.int32(side)] , True, (0, 250, 200), 2)
            cv2.fillConvexPoly(frame, np.int32(side), (255, 255, 255))

        for i, j in self.edges:
            (x0, y0), (x1, y1) = verts[i], verts[j]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 0), 2)
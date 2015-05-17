#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import numpy as np

from tracker import Tracker



class ARTracker(Tracker):

    def __init__(self, src, type="ORB", model="BOX"):
        Tracker.__init__(self, src, type)
        self.model = model

    def draw_3dbox(self, frame, quad):

        ar_box_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                           [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])

        ar_box_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7)]

        ar_box_sides = [(0, 1, 5, 4), (0, 4, 7, 3), (2, 3, 7, 6), (4, 5, 6, 7), (1, 2, 6, 5)]

        x0, y0, x1, y1 = self.target.shape
        quad_3d = np.float32([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])

        h, w = frame.shape[:2]
        cameraMatrix = np.float64([[w,    0,    0.5*(w-1)],
                                   [0,    w,    0.5*(h-1)],
                                   [0,    0,    1        ]])

        dist_coef = np.zeros(4)
        ret, rvec, tvec = cv2.solvePnP(quad_3d, quad, cameraMatrix, dist_coef)
        verts = ar_box_verts * [(x1-x0), (y1-y0), -(x1-x0)*0.5] + (x0, y0, 0)
        verts = cv2.projectPoints(verts, rvec, tvec, cameraMatrix, dist_coef)[0].reshape(-1, 2)

        for i, j, k, g in ar_box_sides:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = verts[i], verts[j], verts[k], verts[g]
            side = (x0, y0), (x1, y1), (x2, y2), (x3, y3)
            # cv2.polylines(frame, [np.int32(side)] , True, (0, 250, 200), 2)
            cv2.fillConvexPoly(frame, np.int32(side), (255, 255, 255))
        for i, j in ar_box_edges:
            (x0, y0), (x1, y1) = verts[i], verts[j]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 0), 2)



    def draw_3dpiramid(self, frame, quad):

        ar_piramid_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0.5, 0.5, 1]])

        ar_piramid_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                            (0, 4), (1, 4), (2, 4), (3, 4)]

        ar_piramid_sides = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (0, 3, 4)]


        x0, y0, x1, y1 = self.target.shape
        quad_3d = np.float32([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])

        h, w = frame.shape[:2]
        cameraMatrix = np.float64([[w,    0,    0.5*(w-1)],
                                   [0,    w,    0.5*(h-1)],
                                   [0,    0,    1        ]])

        dist_coef = np.zeros(4)
        ret, rvec, tvec = cv2.solvePnP(quad_3d, quad, cameraMatrix, dist_coef)
        verts = ar_piramid_verts * [(x1 - x0), (y1 - y0), -(x1 - x0) * 0.5] + (x0, y0, 0)
        verts = cv2.projectPoints(verts, rvec, tvec, cameraMatrix, dist_coef)[0].reshape(-1, 2)

        for i, j, k in ar_piramid_sides:
            (x0, y0), (x1, y1), (x2, y2) = verts[i], verts[j], verts[k]
            side = (x0, y0), (x1, y1), (x2, y2)
            # cv2.polylines(frame, [np.int32(side)] , True, (0, 250, 200), 2)
            cv2.fillConvexPoly(frame, np.int32(side), (255, 255, 255))
        for i, j in ar_piramid_edges:
            (x0, y0), (x1, y1) = verts[i], verts[j]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 0), 2)


    def draw_model(self, frame, quad):
        if self.model == "PIRAMID":
            self.draw_3dpiramid(frame, quad)
        elif self.model == "BOX":
            self.draw_3dbox(frame, quad)
        else:
            self.draw_3dbox(frame, quad)


    def run(self):
        while True:

            ret, frame = self.cap.read()
            if ret is None:
                break

            p1, quad = self.track(frame)
            if p1 is not None and quad is not None:
                self.draw_model(frame, quad)

            cv2.imshow("Augment Reality Window", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break


######################################################

if __name__ == '__main__':
    import sys
    try:
        img_src = sys.argv[1]
    except:
        img_src = "../target_1.png"

    ARTracker(img_src).run()

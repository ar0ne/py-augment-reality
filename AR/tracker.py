#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import cv2


FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6

flann_params_ORB = dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 12,
                   key_size = 20,
                   multi_probe_level = 2)
search_params_ORB = {}

flann_params_SIFT = dict(algorithm = FLANN_INDEX_KDTREE,
                         trees = 5)
search_params_SIFT = dict(checks=100)

flann_params_SURF = dict(algorithm = FLANN_INDEX_KDTREE,
                         trees = 5)
search_params_SURF = dict(checks=100)


MIN_MATCH_COUNT = 10



class Target:
    def __init__(self, shape, keypoints, descriptors):
        self.shape = shape
        self.keypoints = keypoints
        self.descrs = descriptors




class Tracker:
    def __init__(self, src, type="ORB"):

        self.detector, self.matcher = self.init_system(type)

        if self.detector is None or self.matcher is None:
            print("Something wrong with type of tracker(allowed only SIFT, SURF and default ORB)")
            exit(2)

        self.cap = cv2.VideoCapture(0)

        self.target = self.init_pattern(src)


    def init_system(self, type):
        return {
            "ORB":  (cv2.ORB_create(nfeatures=3000),
                     cv2.FlannBasedMatcher(flann_params_ORB,  search_params_ORB)),

            "SIFT": (cv2.xfeatures2d.SIFT_create(nfeatures=3000),
                     cv2.FlannBasedMatcher(flann_params_SIFT, search_params_SIFT)),

            "SURF": (cv2.xfeatures2d.SURF_create(),
                    cv2.FlannBasedMatcher(flann_params_SURF, search_params_SURF))
        }.get(type, "ORB")


    def init_pattern(self, img_src):

        img = cv2.imread(img_src, 0)

        if img is None:
            print("Error: pattern image not found - " + img_src)
            exit(1)

        keypoints, descrs = self.detector.detectAndCompute(img, None)
        if descrs is None:
            descrs = []

        self.matcher.add([descrs])

        return Target((0, 0, img.shape[0], img.shape[1]), keypoints, descrs)


    def track(self, frame):

        self.frame_points, self.frame_descrs = self.detector.detectAndCompute(frame, None)
        if self.frame_descrs is None:
            self.frame_descrs = []
        if len(self.frame_points) < MIN_MATCH_COUNT:
            return None, None
        matches = self.matcher.knnMatch(self.frame_descrs, k = 2)

        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                good.append(m[0])

        if len(good) < MIN_MATCH_COUNT:
            return None, None

        p0 = [self.target.keypoints[m.trainIdx].pt for m in good]

        p1 = [self.frame_points[m.queryIdx].pt for m in good]

        p0, p1 = np.float32((p0, p1))

        M, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)

        if status is None:
            return None, None

        status = status.ravel() != 0
        if status.sum() < MIN_MATCH_COUNT:
            return None, None

        p1 = p1[status]

        x0, y0, x1, y1 = self.target.shape

        quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), M).reshape(-1, 2)

        return p1, quad


    def run(self):
        while True:

            ret, frame = self.cap.read()
            if ret is None:
                break

            p1, quad = self.track(frame)
            if p1 is not None and quad is not None:

                cv2.polylines(frame, [np.int32(quad)], True, (0, 250, 200), 2)
                for (x, y) in np.int32(p1):
                    cv2.circle(frame, (x, y), 2, (255, 255, 255))

            cv2.imshow("Tracker Window", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break


######################################################


if __name__ == '__main__':
    import sys
    try:
        img_src = sys.argv[1]
    except:
        img_src = "../target_1.png"

    Tracker(img_src).run()

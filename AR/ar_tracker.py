#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import numpy as np

from tracker import Tracker
from models import BoxModel


class ARTracker(Tracker):
    def __init__(self, src, type="ORB", model=None):
        Tracker.__init__(self, src, type)
        self._model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value


    def run(self):
        """Main loop. Read frame from camera and try to find target on it. Then draw 3D model."""
        while True:

            ret, frame = self._cap.read()
            if ret is None:
                break

            points, quad = self.track(frame)  # get features and quad of target image
            if points is not None and quad is not None:
                self.model.draw(frame, quad)

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

    ARTracker(img_src, model=BoxModel()).run()

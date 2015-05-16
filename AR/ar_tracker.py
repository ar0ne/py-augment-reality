#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Serj Ar[]ne Shalygailo
"""


import cv2
import numpy as np


from tracker import Tracker



class ARTracker(Tracker):
    def __int__(self, src):
        Tracker.__init__(self, src)

    def run(self):
        pass



if __name__ == '__main__':
    import sys
    try:
        img_src = sys.argv[1]
    except:
        img_src = "target_1.png"

    ARTracker(img_src).run()

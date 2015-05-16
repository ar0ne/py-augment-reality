#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Serj Ar[]ne Shalygailo
email: serj.ar0ne@gmail.com
"""


import AR

if __name__ == "__main__":
    import sys

    try:
        img_src = sys.argv[1]
    except:
        img_src = 'target_1.png'
        print("[INFO] Open default image")
        
        
    AR.Tracker(img_src).track()


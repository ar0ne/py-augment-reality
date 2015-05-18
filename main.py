#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Serj Ar[]ne Shalygailo
email: serj.ar0ne@gmail.com
"""


from AR import Tracker, ARTracker
import argparse
import sys


def create_parser():
    parser = argparse.ArgumentParser(prog="main.py",
                                     description='''This is example of Augment Reality and Tracker application.
                                     Based on Python and OpenCV library''',
                                     epilog='''(c) Serj Ar[]ne 2015 ''')

    parser.add_argument('-a', '--app',   choices=['TRACKER','AR'], default='AR', help="Type of application - just tracker or augment reality.")
    parser.add_argument('-f', '--file',  default="target_1.png", type=argparse.FileType(), help="Name of target image.")
    parser.add_argument('-t', '--type',  choices=['ORB','SIFT', 'SURF'], default='ORB', help="Method of feature detection.")
    parser.add_argument('-m', '--model', choices=['BOX', 'PYRAMID'], default="BOX", help="The model that will be added.")

    return parser


if __name__ == "__main__":
    parser = create_parser()
    namespace = parser.parse_args(sys.argv[1:])

    if namespace.app == "TRACKER":
        Tracker(namespace.file.name, namespace.type).run()
    elif namespace.app == "AR":
        ARTracker(namespace.file.name, namespace.type, namespace.model).run()
    else:
        print("Something goes wrong!")


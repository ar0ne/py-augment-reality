__version__ = "1.0"


from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import numpy as np

from AR import ARTracker
from AR.models import *


class CamApp(App):

    def build(self):
        self.img1 = Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)

        self.capture = cv2.VideoCapture(0)

        self.ar_tracker = ARTracker("../target_2.png",model=BoxModel())

        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def update(self, dt):
        ret, frame = self.capture.read()

        if ret is not None:
            points, quad = self.ar_tracker.track(frame)
            if points is not None and quad is not None:
                self.ar_tracker.model.draw(frame, quad)

            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            self.img1.texture = texture1
        else:
            print("No frame read.")

if __name__ == '__main__':
    CamApp().run()


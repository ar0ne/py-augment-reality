__author__ = 'ar1'


class Model:
    def __init__(self):
        self._verts = None
        self._edges = None
        self._sides = None

    @property
    def verts(self):
        return self._verts

    @verts.setter
    def verts(self, value):
        self._verts = value

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, value):
        self._edges = value

    @property
    def sides(self):
        return self._sides

    @sides.setter
    def sides(self, value):
        self._sides = value


    def draw(self, frame, quad):
        """Draw 3D Model"""
        pass

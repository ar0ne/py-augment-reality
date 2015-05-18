class Target:
    def __init__(self, shape, keypoints, descriptors):
        self._shape = shape
        self._keypoints = keypoints
        self._descrs = descriptors

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def keypoints(self):
        return self._keypoints

    @keypoints.setter
    def keypoints(self, value):
        self._keypoints = value

    @property
    def descrs(self):
        return self._descrs

    @descrs.setter
    def descrs(self, value):
        self._descrs = value

import itertools
import numpy as np
import cv2

class SlamPoint(object):
    """SlamPoint class"""
    newid = itertools.count().__next__
    def __init__(self, point, color):
        self.id = SlamPoint.newid()
        self.point, self.color = np.array(point), color
        self.keypoint_idxs = []
        self.frames = []

    def addObservation(self, frame, keypoint_idx):
        """append to the frames list in which frames the point has been observed at idx=keypoint_idx

        :frame: TODO
        :returns: TODO

        """
        self.keypoint_idxs.append(keypoint_idx)
        self.frames.append(frame)

class SlamMap(object):
    """SlamMap class"""
    def __init__(self):
        self.frames = []
        self.points = []
        self.max_frame_id = 0
        self.max_points_id = 0

    def addPoint(self, point):
        """Add 3D point to the map

        :poin: TODO
        :returns: TODO

        """
        self.points.append(point)
        self.max_points_id += 1
        return self.max_points_id

    def addFrame(self, frame):
        """Add frame to the map

        :arg1: TODO
        :returns: TODO

        """
        self.frames.append(frame)
        self.max_frame_id += 1
        return self.max_frame_id



import itertools
import numpy as np
import cv2
import helpers as helper
import optimizer

class SlamPoint(object):
    """SlamPoint class"""
    newid = itertools.count().__next__
    def __init__(self, point, color):
        self.id = SlamPoint.newid()
        self.point, self.color = np.array(point), color
        self.keypoint_idxs = []
        self.frames = []

    def orb(self):
        """TODO: Docstring for orb.
        :returns: TODO

        """
        return [f.descriptors[key_idx] for f,key_idx in zip(self.frames, self.keypoint_idxs)]
    def orb_distance(self, descriptor):
        """TODO: Docstring for orb_distance.

        :descriptor: TODO
        :returns: TODO

        """
        return min([helper.hamming_distance(o, descriptor) for o in self.orb()])
    def addObservation(self, frame, keypoint_idx):
        """append to the frames list in which frames the point has been observed at idx=keypoint_idx

        :frame: TODO
        :returns: TODO

        """
        if frame.map_points[keypoint_idx] is not None:
            print(frame.map_points[keypoint_idx].point)
        assert frame.map_points[keypoint_idx] is None
        assert frame not in self.frames
        frame.map_points[keypoint_idx] = self
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

    # By default optimize only the last 20 frames
    def optimize(self, fix_points=False, local_window=20, iterations=20):
        """optimize the map
        :fix_points: TODO
        :returns: TODO

        """
        reproj_error = optimizer.optimize(self.frames, self.points, local_window, fix_points, verbose=False, iterations=20)
        if fix_points == False:
            for p in self.points:
                # If an old point was observed only in fewer than 4 frames
                if p.frames[-1].id < self.max_frame_id-5 and len(p.frames) < 3:
                    self.points.remove(p)

        return reproj_error

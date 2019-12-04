import os
import sys
import numpy as np
np.set_printoptions(suppress=True)
import time
import cv2

import helpers as helper
import frame as fm
from display import Display
from slam_map import SlamMap, SlamPoint



class Slam(object):
    """Main functional class containing the slamMap"""
    def __init__(self, K, width, height):
        self.slam_map = SlamMap()
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.width = width
        self.height = height

    def matchFrame(self, frame1, frame2):
        """TODO: Docstring for matchFrame.
        :returns: TODO

        """
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(frame1.descriptors, frame2.descriptors, k=2)
        ratio = 0.8
        match_idx1, match_idx2 = [], []
        frame1_pts, frame2_pts = [], []
        for m1, m2 in matches:
            if (m1.distance < ratio*m2.distance) and (m1.distance < 32):
                match_idx1.append(m1.queryIdx)
                match_idx2.append(m1.trainIdx)

        # We need at least 8 matches for calculating Fundamental matrix
        assert(len(match_idx1) >= 8)
        assert(len(match_idx2) >= 8)

        match_idx1, match_idx2 = np.asarray(match_idx1), np.asarray(match_idx2)
        frame1_pts = fm.normalizePoints(frame1.keypoints[match_idx1, :], self.width, self.height)
        frame2_pts = fm.normalizePoints(frame2.keypoints[match_idx2, :], self.width, self.height)


        E, inlier_mask = cv2.findEssentialMat(frame1_pts, frame2_pts, self.K, method=cv2.RANSAC)
        inlier_mask = inlier_mask.astype(bool).squeeze()
        print("Matches: matches = {}, inliers = {}".format(len(matches), inlier_mask.shape[0]))
        print("Essential matrix: \n{}\n".format(E))

        _, R, t, _ = cv2.recoverPose(E, frame1_pts, frame2_pts, self.K)

        pose = np.eye(4)
        pose[0:3, 0:3] = R
        pose[0:3, 3] = t.squeeze()
        pose = np.linalg.inv(pose)
        print("Pose: \n{}\n".format(pose))

        return pose, match_idx1[inlier_mask], match_idx2[inlier_mask], frame1_pts, frame2_pts

    def searchByProjection(self, curr_frame):
        """TODO: Docstring for search.
        :returns: TODO

        """
        proj_points, mask = helper.projectPoints(self.slam_map.points, curr_frame.pose, self.K, self.width, self.height)

        # obtain projection point descriptors for matching
        proj_keypoints = [cv2.KeyPoint(x=proj_point[0], y=proj_point[1], _size=10) for proj_point in proj_points]
        print(len(proj_keypoints))

        orb = cv2.ORB_create(1000)
        _, proj_descriptors = orb.compute(curr_frame.image, proj_keypoints)
        # index_params= dict(algorithm = 6, table_number = 6, key_size = 12, multi_probe_level = 1)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(proj_descriptors, curr_frame.descriptors, k=2)
        ratio = 0.85
        proj_matches = []
        for m1, m2 in matches:
            if (m1.distance < ratio*m2.distance) and (m1.distance < 64):
                proj_matches.append(m1.trainIdx)

        for i, slam_point in enumerate(self.slam_map.points):
            point = slam_point.point
            # if the proj_point is not inlier
            if not mask[i]:
                continue
            if curr_frame in slam_point.frames:
                continue
            # if this map point is matched with frame
            for m_idx in proj_matches:
                if curr_frame.map_points[m_idx] is None:
                    curr_frame.addMapPoint(point, m_idx)
                    slam_point.addObservation(curr_frame, m_idx)
        return proj_matches

    def processFrame(self, image):
        """

        :image: TODO
        :slam_map: TODO
        :returns: TODO

        """
        start_time = time.time()
        frame = fm.Frame(image, self.K)
        self.slam_map.addFrame(frame)

        if frame.id == 0:
            return

        curr_frame = self.slam_map.frames[-1]
        prev_frame = self.slam_map.frames[-2]

        relative_pose, match_idx1, match_idx2, curr_frame_pts, prev_frame_pts = self.matchFrame(curr_frame, prev_frame)

        # Compose the pose of the frame (this forms our pose estimate for optimize)
        curr_frame.pose = relative_pose @ prev_frame.pose
        P1 = prev_frame.pose[0:3, :]
        P2 = curr_frame.pose[0:3, :]

        # If the map contains points
        good_pts = []
        if len(self.slam_map.points) > 0:
            proj_matches = self.searchByProjection(curr_frame)

        # Triangulate only those points that were not found by projection
        points3d, err = helper.triangulate(P1, prev_frame_pts, P2, curr_frame_pts)
        remaining_point_mask = np.array([curr_frame.map_points[i] is None for i in match_idx1])
        points3d = points3d[remaining_point_mask, :]


        print("Reprojection error before optimize: {}\n".format(err))

        for i in range(points3d.shape[0]):
            color = image[int(np.round(curr_frame.keypoints[match_idx1[i]][1])), int(np.round(curr_frame.keypoints[match_idx1[i]][0]))]
            point = SlamPoint(points3d[i], color)
            self.slam_map.addPoint(point)
        print("Points in map: {}".format(len(self.slam_map.points)))
        print("Time:    {:.2f} ms".format((time.time()-start_time)*1000.0))
        curr_frame_pts_dnm, prev_frame_pts_dnm = fm.denormalizePoints(curr_frame_pts, self.width, self.height), fm.denormalizePoints(prev_frame_pts, self.width, self.height)
        return frame.drawFrame(curr_frame_pts_dnm, prev_frame_pts_dnm)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(" Usage: main.py '<video-file>'");
        exit(-1)

    v_capture = cv2.VideoCapture(sys.argv[1])

    if v_capture.isOpened():
        width   = int(v_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(v_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps     = v_capture.get(cv2.CAP_PROP_FPS)
        no_frames= int(v_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        print(width, height, fps)

        if width > 1024:
            aspect = 1024/width
            height = int(height * aspect)
            width  = 1024

        print(width, height, fps)

    focal_length = 1
    if len(sys.argv) >= 3:
        focal_lengt = int(sys.argv[2])

    # TODO: intrinsic parameters for the camera (better way)?
    K = np.array([[focal_length, 0.0, width//2], [0.0, focal_length, height//2], [0.0, 0.0, 1.0]])

    # Initialize Slam instance
    slam = Slam(K, width, height)
    disp = Display()

    i = 0
    while v_capture.isOpened():
        ret, frame = v_capture.read()

        if ret == True:
            print("---------- Frame {} / {} ----------".format(i, no_frames))
            frame = cv2.resize(frame, (width, height))
            processed_frame = slam.processFrame(frame)
            if i > 0:
                cv2.imshow('frame', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            disp.update(slam.slam_map)
        i += 1

v_capture.release()
cv2.destroyAllWindows()

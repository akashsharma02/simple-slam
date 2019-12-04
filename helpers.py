import numpy as np
import frame as fm

def normalizePoints(points2d, width, height):
    """ normalize the keypoints of the frame w.r.t size of the frame

    :points: TODO
    :returns: TODO

    """
    M = max(width, height)
    no_points = points.shape[0]
    T = np.array([[1/M, 0.0, 0.0],
                  [0.0, 1/M, 0.0],
                  [0.0, 0.0     , 1.0]])
    homo_pts = np.hstack((points, np.ones((no_points, 1))))
    return (T @ homo_pts.T).T[:, 0:2]

def denormalizePoints(points2d, width, height):
    """Denormalize the keypoints of the frame

    :points: TODO
    :width: TODO
    :height: TODO
    :returns: TODO

    """
    M = max(width, height)
    no_points = points.shape[0]
    T = np.array([[1/M, 0.0, 0.0],
                  [0.0, 1/M, 0.0],
                  [0.0, 0.0     , 1.0]])
    Tinv = np.linalg.inv(T)
    homo_pts = np.hstack((points, np.ones((no_points, 1))))
    return (Tinv @ homo_pts.T).T[:, 0:2]

def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    no_points = pts1.shape[0]
    p1_c1 = C1[0, :][:, None]
    p2_c1 = C1[1, :][:, None]
    p3_c1 = C1[2, :][:, None]

    p1_c2 = C2[0, :][:, None]
    p2_c2 = C2[1, :][:, None]
    p3_c2 = C2[2, :][:, None]

    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]

    W = np.zeros((no_points, 4))
    for i in range(no_points):
        A = np.zeros((4, 4))
        A[0, :] = y1[i]*p3_c1.T - p2_c1.T
        A[1, :] = p1_c1.T - x1[i]*p3_c1.T

        A[2, :] = y2[i]*p3_c2.T - p2_c2.T
        A[3, :] = p1_c2.T - x2[i]*p3_c2.T

        U, s, Vt = np.linalg.svd(A)
        W[i, :] = (Vt[-1, :]/Vt[-1, -1])

    #Calculate the reprojection error
    pts1_proj = (C1 @ W.T).T
    pts2_proj = (C2 @ W.T).T
    pts1_proj = (pts1_proj/pts1_proj[:, -1][:, None])[:, 0:2]
    pts2_proj = (pts2_proj/pts2_proj[:, -1][:, None])[:, 0:2]
    err = np.sum(np.square(pts1 - pts1_proj) + np.square(pts2 - pts2_proj))

    return W[:, 0:3], err

def projectPoints(points3d, pose, K, width, height):
    """Project the points through projection matrix

    :points3d: TODO
    :R: TODO
    :t: TODO
    :width: TODO
    :height: TODO
    :returns: TODO

    """
    points3d = np.asarray([point3d.point for point3d in points3d])
    P = pose[0:3, :]

    points4d = np.hstack((points3d, np.ones((points3d.shape[0], 1))))
    pts_proj = (P @ points4d.T).T
    pts_proj = (pts_proj/pts_proj[:, -1][:, None])[:, 0:2]
    pts_proj = fm.denormalizePoints(pts_proj, width, height)

    mask1 = np.logical_and((pts_proj[:, 0] > 0), (pts_proj[:, 0] < height))
    mask2 = np.logical_and((pts_proj[:, 1] > 0), (pts_proj[:, 1] < width))
    mask = np.logical_and(mask1, mask2)
    return pts_proj, mask

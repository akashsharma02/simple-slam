import numpy as np
import frame as fm

def camera2(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    return M2s

def hamming_distance(a, b):
    r = (1 << np.arange(8))[:,None]
    return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)


def recoverPose(E, pts1, pts2, K, width, height):
    '''
    Estimate all possible M2 and return the correct M2 and 3D points P
    :param pred_pts1:
    :param pred_pts2:
    :param intrinsics:
    :param M:
    :return: M2, the extrinsics of camera 2
                     C2, the 3x4 camera matrix
                     P, 3D points after triangulation (Nx3)
    '''

    M1 = np.eye(3)
    C1 = K @ np.hstack((M1, np.zeros((3, 1))))
    M2_list = camera2(E)
    P = []
    M2 = []
    min_negative_count = np.inf
    for i in range(M2_list.shape[-1]):
        M2_inst = M2_list[:, :, i]
        C2_inst = K @ M2_inst

        W_inst, err = triangulate(C1, pts1, C2_inst, pts2)
        negative_count = np.sum(W_inst[:, -1] < 0)
        if np.min(W_inst[:, -1]) > 0:
            P = W_inst
            M2 = M2_inst
    C2 = K @ M2
    R = M2[0:3, 0:3]
    t = M2[0:3, 3]
    return R, t, P

def normalizePoints(points, Kinv):
    """ normalize the keypoints of the frame w.r.t size of the frame

    :points: TODO
    :returns: TODO

    """
    no_points = points.shape[0]
    homo_pts = np.hstack((points, np.ones((no_points, 1))))
    return (Kinv @ homo_pts.T).T[:, 0:2]
def denormalizePoints(points, K):
    """denormalize the keypoints

    :points: TODO
    :K: TODO
    :returns: TODO

    """
    no_points = points.shape[0]
    homo_pts = np.hstack((points, np.ones((no_points, 1))))
    return (K @ homo_pts.T).T[:, 0:2]
def triangulate(C1, pts1, C2, pts2):
    """TODO: Docstring for triangulate.

    :C1: TODO
    :pts1: TODO
    :C2: TODO
    :pts2: TODO
    :returns: 3D world coordinates, ReprojectionError

    """
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

def projectPoints(points3d, pose, width, height):
    """Project the points through projection matrix

    :points3d:
    :pose: TODO
    :width: TODO
    :height: TODO
    :returns: Normalized image coordinates

    """
    points3d = np.asarray(points3d)
    if len(points3d.shape) == 1:
        points3d = points3d[..., np.newaxis].T
    P = pose[0:3, :]

    points4d = np.hstack((points3d, np.ones((points3d.shape[0], 1))))
    pts_proj = (P @ points4d.T).T
    pts_proj = (pts_proj/pts_proj[:, -1][:, None])[:, 0:2]

    mask1 = np.logical_and((pts_proj[:, 0] > 0), (pts_proj[:, 0] < height))
    mask2 = np.logical_and((pts_proj[:, 1] > 0), (pts_proj[:, 1] < width))
    mask = np.logical_and(mask1, mask2)
    return pts_proj, mask

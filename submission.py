"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import cv2
import scipy
import scipy.stats as st
import helper


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    no_points = pts1.shape[0]
    T = np.array([[1/M, 0.0, 0.0],
                  [0.0, 1/M, 0.0],
                  [0.0, 0.0, 1.0]])

    # transform the homogeneous coordinates by M
    homo_pts1 = np.hstack((pts1, np.ones((no_points, 1))))
    homo_pts2 = np.hstack((pts2, np.ones((no_points, 1))))
    scaled_homo_pts1 = (T @ homo_pts1.T).T
    scaled_homo_pts2 = (T @ homo_pts2.T).T

    # Obtain the A matrix
    A_pts1 = np.tile(scaled_homo_pts1, 3)
    A_pts2 = np.hstack((np.tile(scaled_homo_pts2[:, 0][:, None], 3), np.tile(scaled_homo_pts2[:, 1][:, None], 3), np.tile(np.ones((no_points, 1)), 3)))
    A = A_pts1 * A_pts2

    U, s, Vt = np.linalg.svd(A)
    # Last column of V
    F = Vt[-1, :].reshape((3, 3))
    F = helper._singularize(F)
    F = helper.refineF(F, scaled_homo_pts1[:, 0:2], scaled_homo_pts2[:, 0:2])
    F = np.transpose(T) @ F @ T

    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    no_points = pts1.shape[0]
    T = np.array([[1/M, 0.0, 0.0],
                  [0.0, 1/M, 0.0],
                  [0.0, 0.0, 1.0]])

    # transform the homogeneous coordinates by M
    homo_pts1 = np.hstack((pts1, np.ones((no_points, 1))))
    homo_pts2 = np.hstack((pts2, np.ones((no_points, 1))))
    scaled_homo_pts1 = (T @ homo_pts1.T).T
    scaled_homo_pts2 = (T @ homo_pts2.T).T

    # Obtain the A matrix
    A_pts1 = np.tile(scaled_homo_pts1, 3)
    # print(A_pts1)
    A_pts2 = np.hstack((np.tile(scaled_homo_pts2[:, 0][:, None], 3), np.tile(scaled_homo_pts2[:, 1][:, None], 3), np.tile(np.ones((no_points, 1)), 3)))
    # print(A_pts2)
    A = A_pts1 * A_pts2

    U, s, Vt = np.linalg.svd(A)

    # Last two columns of V, reshaped
    F1 = Vt[-1, :].reshape((3, 3)); F2 = Vt[-2, :].reshape((3, 3))

    # Obtain the last constraint the coefficients for the cubic polynomial
    eq = lambda alpha: np.linalg.det(alpha * F1 + (1-alpha)*F2)
    a0 = eq(0)
    a1 = 2*(eq(1)-eq(-1))/3 - (eq(2)-eq(-2))/12
    a2 = 0.5*eq(1) + 0.5*eq(-1) - a0
    a3 = eq(1) - a2 - a1 - a0

    roots = np.roots([a3, a2, a1, a0])
    # Retain only roots which are real
    roots = roots[np.abs(np.imag(roots)) < 1e-8]
    roots = np.real(roots)

    Farray = [root_i*F1 + (1-root_i)*F2 for root_i in roots]
    Farray = [helper._singularize(F) for F in Farray]
    Farray = [np.transpose(T) @ F @ T for F in Farray]

    return Farray

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    return K2.T @ F @ K1


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
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

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    x1 = np.int(np.round(x1))
    y1 = np.int(np.round(y1))
    window_size = 41
    window_offset = window_size // 2
    sigma = 3
    search_length = 41

    # Create a gaussian kernel of window size
    x = np.linspace(-sigma, sigma, window_size+1)
    gkern1d = np.diff(st.norm.cdf(x))
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d = gkern2d/gkern2d.sum()
    gkern2d = np.dstack((gkern2d, gkern2d, gkern2d))

    # Create an image patch around x1, y1
    patch_im1 = im1[y1-window_offset:y1+window_offset+1, x1-window_offset:x1+window_offset+1]

    # Get the epipolar line in im2 for x1, y1
    epi_line = F @ np.array([x1, y1, 1]).reshape((3, 1))
    a = epi_line[0]; b = epi_line[1]; c = epi_line[2]

    y2 = np.arange(y1-search_length, y1+search_length)
    x2 = np.round((-b*y2-c)/a).astype(int)
    valid_pts = (x2 >= window_offset) & (x2 < im2.shape[1] - window_offset) & (y2 >= window_offset) & (y2 < im2.shape[0] - window_offset)
    x2, y2 = x2[valid_pts], y2[valid_pts]

    min_dist = float('inf')
    x2_min = 0; y2_min = 0
    for i in range(x2.shape[0]):
        patch_im2 = im2[y2[i]-window_offset:y2[i]+window_offset+1, x2[i]-window_offset:x2[i]+window_offset+1]
        dist = np.sum(np.square(patch_im1 - patch_im2)*gkern2d)
        if dist < min_dist:
            min_dist = dist
            x2_min, y2_min = x2[i], y2[i]

    return x2_min, y2_min
'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    no_points = pts1.shape[0]
    max_iterations = 1000
    threshold = 0.8

    homo_pts1 = np.hstack((pts1, np.ones((no_points, 1))))
    homo_pts2 = np.hstack((pts2, np.ones((no_points, 1))))

    F_best = np.zeros((3, 3))
    best_inliers = np.zeros((no_points))
    for i in range(max_iterations):
        chosen_points = np.random.randint(0, pts1.shape[0], 7)
        seven_pts1 = pts1[chosen_points, :]; seven_pts2 = pts2[chosen_points, :]
        Farray = sevenpoint(seven_pts1, seven_pts2, M)

        for F in Farray:
            epi_lines = (F @ homo_pts1.T).T
            a = epi_lines[:, 0]; b = epi_lines[:, 1]; c = epi_lines[:, 2]

            dist = homo_pts2 * epi_lines
            dist = dist / np.sqrt(np.square(a[:, None]) + np.square(b[:, None]))
            dist = np.sum(dist, axis=-1)
            inliers = np.where(np.abs(dist) < threshold, True, False)
            no_inliers = np.sum(inliers)
            if no_inliers > np.sum(best_inliers):
                best_inliers = inliers
                F_best = F

    #I observe that refining the solution using the inliers messes the fundamental matrix
    return F_best, best_inliers
'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r)

    if theta == 0:
        return np.eye(3)
    else:
        u = r/theta
        u_cross = np.array([[0.0, -u[2, 0], u[1, 0]], [u[2, 0], 0.0, -u[0, 0]], [-u[1, 0], u[0, 0], 0.0]])
        R = np.eye(3) + np.sin(theta)*u_cross + (1-np.cos(theta))*(u@u.T - np.sum(np.square(u)*np.eye(3)))
        return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    tolerance = 1e-15
    A = (R - R.T)/2
    rho = np.array(A[[2, 0, 1], [1, 2, 0]])[:, None]
    s = np.float(np.linalg.norm(rho))
    c = np.float((np.trace(R) - 1)/2)
    if s < tolerance and (c - 1) < tolerance:
        r = np.array([0.0, 0.0, 0.0])[:, None]
        return r
    elif s < 1e-15 and (c + 1) < tolerance:
        # find non-zero column of R+I
        v = None
        for i in range(R.shape[-1]):
            v = (R + np.eye(3))[:, i]
            if np.count_nonzero(v) > 0:
                break
        u = v/np.linalg.norm(v)
        r = (u*np.pi)[:, None]
        if np.linalg.norm(r) == np.pi and (r[0, 0] == r[1, 0] == 0 and r[2, 0] < 0.0) or (r[0, 0] == 0 and r[1, 0] < 0) or (r[0, 0] < 0):
            return -r
        else:
            return r
    else:
        u = rho/s
        theta = np.arctan2(s, c)
        r = u*theta
        return r

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    w, r2, t2 = x[:-6], x[-6:-3], x[-3:]

    W = w.reshape((w.shape[0] // 3, 3))
    r2 = r2[:, None]
    t2 = t2[:, None]
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    C1 = K1 @ M1
    C2 = K2 @ M2
    homo_W = np.hstack((W, np.ones((W.shape[0], 1))))
    p1_hat = (C1 @ homo_W.T).T
    p2_hat = (C2 @ homo_W.T).T
    p1_hat = (p1_hat/p1_hat[:, -1][:, None])[:, 0:2]
    p2_hat = (p2_hat/p2_hat[:, -1][:, None])[:, 0:2]

    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])
    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    x_init = P_init.flatten()
    R2 = M2_init[:, 0:3]
    t2 = M2_init[:, 3]

    r2 = invRodrigues(R2)
    x_init = np.append(x_init, r2.flatten())
    x_init = np.append(x_init, t2.flatten())

    func = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)
    x_opt, _ = scipy.optimize.leastsq(func, x_init)

    w_opt, r2_opt, t2_opt = x_opt[:-6], x_opt[-6:-3], x_opt[-3:]
    W_opt = w_opt.reshape((w_opt.shape[0] // 3, 3))
    r2_opt = r2_opt[:, None]
    t2_opt = t2_opt[:, None]

    R2_opt = rodrigues(r2_opt)
    M2_opt = np.hstack((R2_opt, t2_opt))

    return M2_opt, W_opt

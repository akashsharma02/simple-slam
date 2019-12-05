import g2o
import numpy as np

def optimize(frames, points, local_window, fix_points=False, verbose=False, iterations=50):
    """optimize using g2o

    :frame: TODO
    :points: TODO
    :local_window: TODO
    :fix_points: TODO
    :verbose: TODO
    :iterations: TODO
    :returns: TODO

    """
    if local_window is None:
        local_frames = frames
    else:
        local_frames = frames[-local_window:]

    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    # add normalized camera
    cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
    cam.set_id(0)
    optimizer.add_parameter(cam)


    huber_threshold = np.sqrt(5.991)
    huber_kernel = g2o.RobustKernelHuber(huber_threshold)
    graph_frames, graph_points = {}, {}

    for f in (local_frames if fix_points else frames):
        pose = f.pose
        se3 = g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3])
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)

        v_se3.set_id(f.id*2)
        v_se3.set_fixed(f.id <=1 or f not in local_frames)
        optimizer.add_vertex(v_se3)

        est = v_se3.estimate()
        assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
        assert np.allclose(pose[0:3, 3], est.translation())


        graph_frames[f] = v_se3

    for p in points:
        if not any([f in local_frames for f in p.frames]):
            continue

        pt = g2o.VertexSBAPointXYZ()
        pt.set_id(p.id*2 + 1)
        pt.set_estimate(p.point)
        pt.set_marginalized(True)
        pt.set_fixed(fix_points)
        optimizer.add_vertex(pt)
        graph_points[p] = pt

        for f, idx in zip(p.frames, p.keypoint_idxs):
            if f not in graph_frames:
                continue
            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_parameter_id(0, 0)
            edge.set_vertex(0, pt)
            edge.set_vertex(1, graph_frames[f])
            edge.set_measurement(f.keypoints[idx])
            edge.set_information(np.eye(2))
            edge.set_robust_kernel(huber_kernel)
            optimizer.add_edge(edge)

    optimizer.set_verbose(verbose)
    optimizer.initialize_optimization()
    optimizer.optimize(iterations)

    for f in graph_frames:
        est = graph_frames[f].estimate()
        R = est.rotation().matrix()
        t = est.translation()
        f.pose = np.eye(4)
        f.pose[0:3, 0:3] = R
        f.pose[0:3, 3] = t

    if not fix_points:
        for p in graph_points:
            p.pt = np.array(graph_points[p].estimate)

    return optimizer.active_chi2()

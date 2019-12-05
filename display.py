from multiprocessing import Process, Queue
import numpy as np
import pangolin
import OpenGL.GL as gl

class Display(object):
    """3D Display using pangolin"""
    def __init__(self):
        self.message = None
        self.q = Queue()
        self.viewer_process = Process(target=self.worker, args=((self.q),))
        self.viewer_process.daemon = True
        self.viewer_process.start()

    def worker(self, q):
        """Worker thread that continuously updates the display

        :q: TODO
        :returns: TODO

        """
        self.init(1280, 1024)
        while True:
            self.refresh(q)

    def init(self, width, height):
        """Initialize the 3D display (pangolin)

        :width: TODO
        :height: TODO
        :returns: TODO

        """
        pangolin.CreateWindowAndBind('Map viewer', width, height)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(width, height, 420, 420, width//2, height//2, 0.2, 10000),
            pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -width/height)
        self.dcam.SetHandler(self.handler)

    def refresh(self, q):
        """Refresh the display if there is new data

        :q: TODO
        :returns: TODO

        """
        while not q.empty():
            self.message = q.get()
        if self.message is not None:
            map_points, poses, colors = self.message

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            self.dcam.Activate(self.scam)

            if poses is not None:
                if poses.shape[0] >= 2:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawCameras(poses[0:-1, :])
                    gl.glColor3f(1.0, 1.0, 0.0)
                    pangolin.DrawCameras(poses[-1:, :])

            if map_points is not None and colors is not None:
                if map_points.shape[0] != 0:
                    gl.glPointSize(3)
                    gl.glColor3f(1.0, 0.0, 0.0)
                    pangolin.DrawPoints(map_points, colors)

        pangolin.FinishFrame()

    def update(self, slam_map):
        """TODO: Docstring for updateDisplay.
        :returns: TODO

        """
        if self.q is None:
            return

        points = np.asarray([point.point for point in slam_map.points])
        poses = np.asarray([np.linalg.inv(frame.pose) for frame in slam_map.frames])
        colors = np.asarray([point.color/256.0 for point in slam_map.points])
        self.q.put((points, poses, colors))

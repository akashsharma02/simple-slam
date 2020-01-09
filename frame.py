import time
import math
import numpy as np
import itertools
import cv2
import helpers as helper



class Frame(object):
    """ Frame object contains keypoints, associated map-points, etc"""
    newid = itertools.count().__next__
    def __init__(self, image, K):
        """ Detect keypoints and compute descriptors"""
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.pose = np.eye(4)
        if image is not None:
            self.height, self.width = image.shape[0:2]
            self.id = Frame.newid()
            self.image = image
            self.extractFeatures(image)
            self.map_points = [None]*len(self.keypoints_un)


    def extractFeatures(self, image):
        """Extract keypoints and compute descriptors

        :image:
        :returns: TODO

        """
        orb = cv2.ORB_create()
        corners = cv2.goodFeaturesToTrack(np.mean(image, axis=2).astype(np.uint8), 3000, qualityLevel=0.02, minDistance=7)
        keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], _size=20) for c in corners]
        keypoints, self.descriptors = orb.compute(image, keypoints)
        keypoints =  [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        self.keypoints_un = np.asarray(keypoints)

    def drawFrame(self, points1, points2):
        """Draw keypoints and tracks on the image

        :points1:
        :points2:
        :returns: TODO

        """
        image = np.copy(self.image)
        if (points1.shape[0] > 0) and (points2.shape[0] > 0):
            for i in range(points1.shape[0]):
                point1 = points1[i, :]
                point2 = points2[i, :]
                cv2.circle(image, (int(point1[0]), int(point1[1])), color=(0, 255, 0), radius=2)
                cv2.arrowedLine(image,(int(point1[0]), int(point1[1])),(int(point2[0]), int(point2[1])), (0, 0, 255), 1, 8, 0, 0.2)
        return image


    def SSC(self, keypoints, num_ret_points, tolerance, cols, rows):
        """ Adaptive non-maximal suppression to sparsify/distribute the keypoints
            Reference: https://github.com/BAILOOL/ANMS-Codes.git
            TODO: Little slow, maybe change approach

        :image: TODO
        :returns: TODO

        """
        exp1 = rows + cols + 2*num_ret_points
        exp2 = 4*cols + 4*num_ret_points + 4*rows*num_ret_points + rows*rows + cols*cols - 2*rows*cols + 4*rows*cols*num_ret_points
        exp3 = math.sqrt(exp2)
        exp4 = (2*(num_ret_points - 1))

        sol1 = -round(float(exp1+exp3)/exp4) # first solution
        sol2 = -round(float(exp1-exp3)/exp4) # second solution

        high = sol1 if (sol1>sol2) else sol2 #binary search range initialization with positive solution
        low = math.floor(math.sqrt(len(keypoints)/num_ret_points))

        prevWidth = -1
        selected_keypoints = []
        ResultVec = []
        result = []
        complete = False
        K = num_ret_points
        Kmin = round(K-(K*tolerance))
        Kmax = round(K+(K*tolerance))

        while(~complete):
            width = low+(high-low)/2
            if (width == prevWidth or low>high): #needed to reassure the same radius is not repeated again
                ResultVec = result #return the keypoints from the previous iteration
                break

            c = width/2; #initializing Grid
            numCellCols = int(math.floor(cols/c));
            numCellRows = int(math.floor(rows/c));
            coveredVec = [ [False for i in range(numCellCols+1)] for j in range(numCellCols+1)]
            result = []

            for i in range(len(keypoints)):
                row = int(math.floor(keypoints[i].pt[1]/c)) #get position of the cell current point is located at
                col = int(math.floor(keypoints[i].pt[0]/c))
                if (coveredVec[row][col]==False): # if the cell is not covered
                    result.append(i)
                    rowMin = int((row-math.floor(width/c)) if ((row-math.floor(width/c))>=0) else 0) #get range which current radius is covering
                    rowMax = int((row+math.floor(width/c)) if ((row+math.floor(width/c))<=numCellRows) else numCellRows)
                    colMin = int((col-math.floor(width/c)) if ((col-math.floor(width/c))>=0) else 0)
                    colMax = int((col+math.floor(width/c)) if ((col+math.floor(width/c))<=numCellCols) else numCellCols)
                    for rowToCov in range(rowMin, rowMax+1):
                        for colToCov in range(colMin, colMax+1):
                            if (~coveredVec[rowToCov][colToCov]):
                                coveredVec[rowToCov][colToCov] = True #cover cells within the square bounding box with width w

                if (len(result)>=Kmin and len(result)<=Kmax): #solution found
                    ResultVec = result
                    complete = True
                elif (len(result)<Kmin):
                    high = width-1 #update binary search range
                else:
                    low = width+1
                prevWidth = width

        for i in range(len(ResultVec)):
            selected_keypoints.append(keypoints[ResultVec[i]])

        return selected_keypoints

    @property
    def keypoints(self):
        if not hasattr(self, '_kps'):
            self._kps = helper.normalizePoints(self.keypoints_un, self.K_inv)
        return self._kps


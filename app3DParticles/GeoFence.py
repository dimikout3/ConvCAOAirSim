import numpy as np
from scipy.spatial import ConvexHull

class GeoFence:

    def __init__(self, *args, **kwargs):

        if 'width' in kwargs:
            self.initializeSquare(**kwargs)
        else:
            self.initializeSphere(**kwargs)


    def initializeSquare(self, **kwargs):
        print(f"Initialize GeoFence as square")

        self.type = 'Square'
        self.centerX = kwargs['centerX']
        self.centerY = kwargs['centerY']

        self.width = kwargs['width']
        self.length = kwargs['length']
        self.height = kwargs['height']

        self.initializeHull()


    def initializeHull(self):

        xHigh, yHigh, zHigh = self.getHighValues()
        xLow, yLow, zLow = self.getLowValues()

        x = np.array([xHigh, xLow])
        y = np.array([yHigh, yLow])
        z = np.array([zHigh, zLow])
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        x = x.reshape(x.size)
        y = y.reshape(y.size)
        z = z.reshape(z.size)
        edges = np.stack((x,y,z), axis=1)

        self.edges = edges
        self.hull = ConvexHull(edges)


    def initializeSphere(self, **kwargs):
        print(f"Initialize GeoFence as sphere")


    def getHighValues(self):

        highX = self.centerX + self.length/2
        highY = self.centerY + self.width/2
        highZ = 0

        return highX, highY, highZ


    def isInside(self, points):

        isInsideList = []
        for ind, point in enumerate(points):
            # new_points = np.append(self.edges, point, axis=0)
            new_points = np.concatenate((self.edges, [point]))
            new_hull = ConvexHull(new_points)
            if list(self.hull.vertices) == list(new_hull.vertices):
                # import pdb; pdb.set_trace()
                isInsideList.append(ind)

        return isInsideList


    def getLowValues(self):

        lowX = self.centerX - self.length/2
        lowY = self.centerY - self.width/2
        lowZ = -(self.height)

        return lowX, lowY, lowZ

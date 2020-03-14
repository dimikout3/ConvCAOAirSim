class Discretizator:

    def __init__(self, discrete=None, geofence=None):

        if discrete == None or geofence == None:
            print("[ERROR] Discretizator need geofence and discrete steps !!")
            return 0

        self.discreteX = discrete['x']
        self.discreteY = discrete['y']
        self.discreteZ = discrete['z']

        self.highX, self.highY, self.highZ = geofence.getHighValues()
        self.lowX, self.lowY, self.lowZ = geofence.getLowValues()
        self.lowValues = np.array([self.lowX, self.lowY, self.lowZ])

        self.stepX = (self.highX - self.lowX)/self.discreteX
        self.stepY = (self.highY - self.lowY)/self.discreteY
        self.stepZ = (self.highZ - self.lowZ)/self.discreteZ
        self.stepSizes = np.array([self.stepX, self.stepY, self.stepZ])


    def report(self):

        print("Discretizator has ")
        print(f"discreteX={self.discreteX} discreteY={self.discreteY} discreteZ={self.discreteZ}")
        print(f"highX={self.highX} highY={self.highY} highZ={self.highZ}")
        print(f"lowX={self.lowX} lowY={self.lowY} lowZ={self.lowZ}")
        print(f"stepX={self.stepX} stepY={self.stepY} stepZ={self.stepZ}")


    def descretize(self, data):

        """Given a set of data [(x,y,z), dtype=float] convert them to descrete"""

        descrete = (data - self.lowValues) / self.stepSizes

        return descrete.astype(np.int)

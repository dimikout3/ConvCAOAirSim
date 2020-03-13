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


    def initializeSphere(self, **kwargs):
        print(f"Initialize GeoFence as sphere")


    def getHighValues(self):

        highX = self.centerX + self.length/2
        highY = self.centerY + self.width/2
        highZ = 0

        return highX, highY, highZ


    def getLowValues(self):

        lowX = self.centerX - self.length/2
        lowY = self.centerY - self.width/2
        lowZ = -(self.height)

        return lowX, lowY, lowZ

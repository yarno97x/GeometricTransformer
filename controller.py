from quaternions import Quaternion
from coordinates import P2, P3
import numpy as np

class Controller:
    def __init__(self):
        self.current2DPoints = []
        self.current3DPoints = []

    def add2DPoint(self, x, y):
        self.current2DPoints.append(P2(x, y))

    def rotationMatrix2D(self, angle):
        angle = np.radians(angle)
        return np.array([[np.cos(angle), -np.sin(angle), 0], 
                         [np.sin(angle),  np.cos(angle), 0],
                         [0,              0,             0]], dtype=np.float16)
    
    def scale2DMatrix(self, s):
        return s * np.identity(3)
    
    def translate2D(self, x, y):
        matrix = np.zeros(shape =(3, 3))
        matrix[0, -1] = x
        matrix[1, -1] = y
        matrix[2, -1] = 1
        return matrix
    
    def getMatrix2D(self, s, angle, x, y):
        scaled_angle = np.matmul(self.scale2DMatrix(s), self.rotationMatrix2D(angle))
        print(scaled_angle)
        return scaled_angle + self.translate2D(x, y)
    
    def pack2D(self):
        package = []
        for coord in self.current2DPoints:
            np_arr = coord.to_homogeneous().reshape(-1, 1)     
            package.append(np_arr)
        return np.hstack(package)
    
    def perform2D(self, scale = 1, angleDegrees = 0, tx = 0, ty = 0):
        matrix = self.getMatrix2D(scale, angleDegrees, tx, ty)
        return np.round(np.matmul(matrix, self.pack2D()), 3)

    def add3DPoint(self, x, y, z):
        self.current3DPoints.append(P3(x, y, z))

    def rotationMatrix3D(self, x, y, z, angle):
        return Quaternion(x, y, z, angle).r
    
    def scale3DMatrix(self, s):
        return s * np.identity(4)
    
    def translate3D(self, x, y, z):
        matrix = np.zeros(shape =(4, 4))
        matrix[0, -1] = x
        matrix[1, -1] = y
        matrix[2, -1] = z
        matrix[3, -1] = 1
        return matrix
    
    def getMatrix3D(self, s, axis, angle, x, y, z):
        scaled = self.scale3DMatrix(s)
        # print(scaled)
        rot = np.zeros((4, 4))
        rot[:3, :3] = self.rotationMatrix3D(*axis, np.deg2rad(angle))
        # print(rot)
        scaled_angle = np.matmul(scaled, rot)
        # print(scaled_angle)
        return scaled_angle + self.translate3D(x, y, z)
    
    def pack3D(self):
        package = []
        for coord in self.current3DPoints:
            np_arr = coord.to_homogeneous().reshape(-1, 1)     
            package.append(np_arr)
        return np.hstack(package)
    
    def perform3D(self, axisX, axisY, axisZ, scale = 1, angleDegrees = 0, tx = 0, ty = 0, tz = 0):
        matrix = self.getMatrix3D(scale, (axisX, axisY, axisZ), angleDegrees, tx, ty, tz)
        return np.round(np.matmul(matrix, self.pack3D()), 3)

if __name__ == "__main__":
    c = Controller()
    c.add2DPoint(1,1) 
    c.add2DPoint(2,1) 
    c.add2DPoint(3,1) 
    c.add2DPoint(4,1) 
    print(c.pack2D())

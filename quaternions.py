import numpy as np

class Quaternion:
    def __init__(self, x = None, y = None, z = None, angle = None, qx = None, qy = None, qz = None, qw = None):
        if x is not None :
            self.to_quaternion(x, y, z, angle)
        else :
            if qw < 0:
                qx *= -1
                qy *= -1
                qz *= -1
                qw *= -1
            self.x = qx
            self.y = qy
            self.z = qx
            self.w = qw
            self.v = np.array([self.x, self.y, self.z], dtype=np.float32)
        self.rotation_matrix()

    def to_quaternion(self, x, y, z, angle):
        vec = np.array([x, y, z], dtype=np.float32)
        nor = np.linalg.norm(vec)
        self.v = vec / nor * np.sin(angle / 2)
        self.x, self.y, self.z = self.v
        self.w = np.round(np.cos(angle / 2), 2)

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z}, {self.w})"
    
    def rotation_matrix(self):
        self.r = np.round(np.array([
            [1 - 2*(self.y ** 2 + self.z ** 2),       2 * (self.x * self.y - self.z * self.w), 2 * (self.x * self.z + self.y * self.w)],
            [2 * (self.x * self.y + self.z * self.w), 1 - 2*(self.x ** 2 + self.z ** 2),       2 * (self.z * self.y - self.x * self.w)],
            [2 * (self.x * self.z - self.y * self.w), 2 * (self.z * self.y + self.x * self.w), 1 - 2*(self.x ** 2 + self.y ** 2)      ]
                         ]), )
        
    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return np.matmul(self.r, other)
        elif isinstance(other, Quaternion):
            newV = np.cross(self.v, other.v) + self.w * other.v + other.w * self.v
            newW = np.array([self.w * other.w - np.dot(self.v, other.v)])
            x, y, z, w = np.concat([newV, newW])
            return Quaternion(qx=x, qy=y, qz=z, qw=w)
        
    def __truediv__(self, other):
        if isinstance(other, Quaternion):
            newV = np.cross(self.v, other.v) + self.w * other.v - other.w * self.v
            newW = np.array([-self.w * other.w - np.dot(self.v, other.v)])
            x, y, z, w = np.concat([newV, newW])
            return Quaternion(qx=x, qy=y, qz=z, qw=w)

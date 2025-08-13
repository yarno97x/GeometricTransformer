import numpy as np

class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def augment(self):
        return self.vec / self.vec[-1]
    
    def to_inhomogeneous(self):
        return np.array([value/self.vec[-1] for value in self.vec[:-1]], dtype=np.float32)
    
    def to_homogeneous(self):
        return self.vec
    
    def __eq__(self, value):
        if isinstance(value, type(self)):
            a1 = self.augment()
            a2 = value.augment()
            return all([a1[i] == a2[i] for i in range(len(self.vec[:-1]))])
        raise TypeError("Type error on equation operator")
    
class P2(Coordinate):
    def __init__(self, x, y, w = 1):
        super().__init__(x, y)
        self.vec = np.array([x, y, w], dtype=np.float32)

    def __rmul__(self, other):
        if isinstance(other, int):
            new_vec = other * self.vec
            new_vec[-1] /= other
            x, y, w = new_vec
        return P2(x, y, w)
    
class P3(Coordinate):
    def __init__(self, x, y, z, w = 1):
        super().__init__(x, y)
        self.z = z
        self.vec = np.array([x, y, z, w], dtype=np.float32)



if __name__ == "__main__":
    c = P3(1,2,2,4)
    c2 = P2(1,2,2)
    print(c == c2)

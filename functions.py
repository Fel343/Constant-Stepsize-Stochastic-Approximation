# Scaling function for general convex objective function 

import numpy as np

class Functions:
    def __init__(self, k, epsilon = 1e-10):
        self.k = k
        self.two_k = 2 * k
        self.epsilon = epsilon
    # Derivative of x^(2k)/(2k)
    def polynomial(self, x):
        x_clipped = np.clip(x, -1e10, 1e10)
        return x**(self.two_k - 1)
    
    # Derivative of x^(2k)/(2k) + sin(x)^(2k)/2k
    def trigonometric(self, x):
        x_clipped = np.clip(x, -1e10, 1e10)
        return x**(self.two_k - 1) + np.sin(x)**(self.two_k - 1) * np.cos(x)
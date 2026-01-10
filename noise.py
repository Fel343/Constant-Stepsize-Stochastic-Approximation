# Different noise distributions with mean 0 and specified variance

import numpy as np

class Noise:
    """
    Noise generator with different distributions, all with mean 0 and specified variance.
    
    Parameters:
        variance: The desired variance for the noise distribution
        n: Number of samples to generate per call
    """
    def __init__(self, variance, n):
        self.mean = 0
        self.variance = variance
        self.std = np.sqrt(variance)  # Standard deviation
        self.n = n
    
    def normal(self):
        """Normal distribution: N(0, variance)"""
        return np.random.normal(self.mean, self.std, self.n)

    def uniform(self):
        """
        Uniform distribution: U(-a, a)
        Variance = (b-a)^2 / 12 = (2a)^2 / 12 = a^2 / 3
        So a = sqrt(3 * variance)
        """
        a = np.sqrt(3 * self.variance)
        return np.random.uniform(-a, a, self.n)
    
    def laplace(self):
        """
        Laplace distribution: Laplace(0, scale)
        Variance = 2 * scale^2
        So scale = sqrt(variance / 2)
        """
        scale = np.sqrt(self.variance / 2)
        return np.random.laplace(self.mean, scale, self.n)

    def student_t(self, df=5):
        """
        Student-t distribution scaled to have specified variance.
        For df > 2, Var(t_df) = df / (df - 2)
        We scale by sqrt(variance * (df - 2) / df) to get desired variance.
        """
        df = -2 * self.variance / (1 - self.variance)
        return  np.random.standard_t(df, self.n)
    



import numpy as np
import math

class Binomial:
  def __init__(self, n, p):
    self.n = n
    self.p = p
  def prob(self, k):
    return math.comb(self.n, k) * self.p**k * (1-self.p)**(self.n-k)

class Multinomial:
    def __init__(self, outcomes_probs):
        self.outcomes_probs = outcomes_probs
    def prob(self, X):
        return self.outcomes_probs[X]

class Poisson:
  def __init__(self, lmbda):
    self.lmbda = lmbda
  def prob(self, x):
    return (self.lmbda ** x) * np.exp(-self.lmbda) / np.math.factorial(x)

class Gaussian:
  def __init__(self, mu, sigma):
    self.mu = mu
    self.sigma = sigma
  def prob(self, x):
    return (1/(self.sigma * np.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - self.mu)/self.sigma)**2)

class Uniform:
  def __init__(self, a, b):
    self.a = a
    self.b = b
  def prob(self, x):
    if self.a > self.b:
      raise Exception("b must be greater than a.")
    elif self.a <= x and x <= self.b:
      return 1 / (self.b - self.a)
    else:
      return 0

class Inverse_Gaussian:
  def __init__(self, lmbda, mu):
    self.lmbda = lmbda
    self.mu = mu
  def inverse_gaussian(self, x):
    return np.sqrt(self.lmbda / (2 * math.pi * x**3)) * np.exp(-((self.lmbda(x - self.mu)**2)/(2 * self.mu**2 * x)))

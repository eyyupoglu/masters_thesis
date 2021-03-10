from QuantLib import *
import numpy as Numpy
import matplotlib.pyplot as Matplotlib

# process = QuantLib 1-dimensional stochastic process object
def generate_paths(process, maturity, nPaths, nSteps):
    generator = UniformRandomGenerator()
    sequenceGenerator = UniformRandomSequenceGenerator(nSteps, generator)
    gaussianSequenceGenerator = GaussianRandomSequenceGenerator(sequenceGenerator)
    paths = Numpy.zeros(shape = ((nPaths), nSteps + 1))
    pathGenerator = GaussianPathGenerator(process, maturity, nSteps, gaussianSequenceGenerator, False)
    for i in range(nPaths):
        path = pathGenerator.next().value()
        paths[i, :] = Numpy.array([path[j] for j in range(nSteps + 1)])
    return paths




import numpy as np


def number_of_networks(N, k):
    return ((2**(2**k) * np.math.factorial(N))/(np.math.factorial(N-k)))**N


print(number_of_networks(15, 3))

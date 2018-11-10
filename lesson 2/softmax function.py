import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    Lexp = np.exp(L)
    Lsum = np.sum(Lexp)
    return [Lexp[i]/Lsum for i in range(len(Lexp))]
    
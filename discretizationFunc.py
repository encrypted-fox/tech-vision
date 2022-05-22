import numpy as np
import matplotlib.pyplot as plt

def discretizationFunc():
    x = np.arange(0, 2*np.pi, 0.05)
    y = np.sin(4*x)+0.5*np.sin(6*x)+0.25*np.sin(8*x)
    plt.plot(2*x, y, '.')
    plt.show()
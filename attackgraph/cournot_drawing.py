import os
from attackgraph import file_op as fp

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

def plot_BR():
    regret_maxent = [4.0, 1.0, 1.0, 0.25, 0.25, 0.0, 0.0]
    regret_minent = [4.0, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0]

    x = np.arange(1, len(regret_maxent) + 1)

    plt.plot(x, regret_maxent, '-ro', label="MaxEnt")
    plt.plot(x, regret_minent , '--go', label="MinEnt")
    plt.xlabel("Epochs")
    plt.ylabel("Regret")
    plt.title("Regret Curves of Cournot Game")
    plt.legend(loc="best")
    plt.show()


def plot_BD():
    regret_maxent = [4.0, 1.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    regret_minent = [4.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    x = np.arange(1, len(regret_maxent) + 1)

    plt.plot(x, regret_maxent, '-ro', label="MaxEnt")
    plt.plot(x, regret_minent , '--go', label="MinEnt")
    plt.xlabel("Epochs")
    plt.ylabel("Regret")
    plt.title("Regret Curves of Cournot Game with Beneficial Deviation")
    plt.legend(loc="best")
    plt.show()

# plot_BR()
plot_BD()
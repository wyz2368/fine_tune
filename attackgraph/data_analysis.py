import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import os

from attackgraph import file_op as fp
from attackgraph import json_op as jp

def expected_payoff(a_BD, att_payoff, d_BD, def_payoff):
    rounds = len(a_BD)
    X = np.linspace(1, rounds, rounds, endpoint=True)
    plt.xticks(np.linspace(1, rounds, rounds, endpoint=True))
    a_BD = np.array(a_BD)
    att_payoff = np.array(att_payoff)
    d_BD = np.array(d_BD)
    def_payoff = np.array(def_payoff)

    abd, = plt.plot(X, a_BD, color="red", linewidth=1.0, linestyle="--")
    apf, = plt.plot(X, att_payoff, color = "green", linewidth = 1.0, linestyle = "-")
    dbd, = plt.plot(X, d_BD, color = "orange", linewidth = 1.0, linestyle = "--")
    dpf, = plt.plot(X, def_payoff, color = "blue", linewidth = 1.0, linestyle = "-")

    plt.legend([abd, apf, dbd, dpf], ['Att.dev.', 'Att.eq.', 'Def.dev.', 'Def.eq.'])

    plt.show()


def learning_curve(data):
    path = os.getcwd() + '/learning_curve/' + data + '.pkl'
    curve = fp.load_pkl(path)
    plt.plot(curve, color=np.random.rand(3,))
    plt.xlabel("Time Steps")
    plt.ylabel("Average Reward over 250 episodes")
    plt.title("Learning Curve")
    plt.show()


def learning_curve_many(data):
    for name in data:
        path = os.getcwd() + '/learning_curve/' + name + '.pkl'
        curve = fp.load_pkl(path)
        plt.plot(curve, color=np.random.rand(3,))
    plt.xlabel("Time Steps")
    plt.ylabel("Average Reward over 250 episodes")
    plt.title("Learning Curve")
    plt.show()
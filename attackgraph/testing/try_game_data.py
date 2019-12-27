# from attackgraph import game_data
# from attackgraph import DagGenerator as dag
import numpy as np
# import random
# import time
# from attackgraph import util
from attackgraph import file_op as fp
import os

path = os.getcwd() + '/game_data/game_rs.pkl'
game = fp.load_pkl(path)
# print(game.nasheq)
# print(game.att_str)
# print(game.def_str)
print(game.payoffmatrix_def)
print('_________________________')
print(game.payoffmatrix_att)
# print(game.regret())
# print(game.mean_regret())


# param = (4, 0.7, 0.286)
# identity = 0
# nasheq = {}
# nasheq[1] = {}
# nasheq[1][0] = np.array([1])
# nasheq[2] = {}
# nasheq[2][0] = np.array([0.2,0.8])
# nasheq[3] = {}
# nasheq[3][0] = np.array([0.3,0.5,0.2])
# nasheq[4] = {}
# nasheq[4][0] = np.array([0.1,0.7,0.1,0.1])
#
# k, gamma, alpha = param
# num_str = len(nasheq.keys())
# delta = nasheq[num_str][identity]
# denom = 0
# for i in np.arange(num_str-1):
#     temp = nasheq[i+1][identity].copy()
#     temp.resize(delta.shape)
#     print(temp)
#     delta += gamma**(num_str-1-i)*temp #TODO: make sure this is correct.
#     denom += gamma**(num_str-1-i)
#
# denom += 1
#
# # print(np.round(delta/denom,2))
# print(delta/np.sum(delta))
# print(np.sum(delta/np.sum(delta)))
# print(delta/denom)
# print(np.sum(delta/denom))
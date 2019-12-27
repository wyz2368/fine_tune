import numpy as np
from attackgraph import file_op as fp
# from attackgraph.uniform_str_init import act_def, act_att
# import math
import os
import copy
import glob
from psutil import virtual_memory
from queue import PriorityQueue as pq
# from attackgraph.cournot import extract_submatrix
from attackgraph.gambit_analysis import do_gambit_analysis

# # import networkx as nx
import random
# # import itertools
import time
import datetime
# import training
# # import tensorflow as tf
# import pickle as pk
#

# a = np.array([[1,0],[0,1]])
# b = np.array([[1,0],[0,1]])
#
# ne_att, ne_def = do_gambit_analysis(a,b, maxent=False, minent=True, return_list=False)
# ne_list = do_gambit_analysis(a,b, maxent=False, minent=False, return_list=True)
#
# print(ne_att, ne_def)
# print(ne_list)
#

a = np.array([0,-1,1,-4])
print(np.where(a < 0)[0])




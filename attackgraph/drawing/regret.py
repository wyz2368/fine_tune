import os
from attackgraph import file_op as fp

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from attackgraph import evaluation_combined as ec


path_def = os.getcwd() + '/drawing/matrix/' + 'payoff_matrix_def.pkl'
path_att = os.getcwd() + '/drawing/matrix/' + 'payoff_matrix_att.pkl'
payoff_matrix_def = fp.load_pkl(path_def)
payoff_matrix_att = fp.load_pkl(path_att)

print(np.shape(payoff_matrix_def))

# payoff_matrix_def = np.delete(payoff_matrix_def, np.arange(51, 81), 0)
# payoff_matrix_def = np.delete(payoff_matrix_def, np.arange(51, 81), 1)
# payoff_matrix_att = np.delete(payoff_matrix_att, np.arange(51, 81), 0)
# payoff_matrix_att = np.delete(payoff_matrix_att, np.arange(51, 81), 1)

child_partition = {'RS': 80, 'SP': 80}
ec.do_evaluation(payoff_matrix_def, payoff_matrix_att, child_partition)

# curves_dict_def, curves_dict_att = ec.formal_regret_curves(payoff_matrix_def, payoff_matrix_att, child_partition)
# save_path = os.getcwd() + '/drawing/matrix/'
# fp.save_pkl(curves_dict_att, save_path + 'curves_dict_att.pkl')
# fp.save_pkl(curves_dict_def, save_path + 'curves_dict_def.pkl')






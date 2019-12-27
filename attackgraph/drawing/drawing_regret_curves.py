import os
from attackgraph import file_op as fp

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import math

load_path = os.getcwd() + '/drawing/matrix/'
curves_dict_att = fp.load_pkl(load_path + 'curves_dict_att.pkl')
curves_dict_def = fp.load_pkl(load_path + 'curves_dict_def.pkl')


method_list = ['RS', 'SP']

x = np.arange(1,81)

plt.plot(x, curves_dict_def['RS'], '-ro', label= "DO")
plt.plot(x, curves_dict_def['SP'], '-go', label= "DO+SP")
plt.xlabel("Epochs")
plt.ylabel("Defender's Regret w.r.t Combined Game")
plt.title("Regret Curves")
plt.legend(loc="best")
plt.show()


# plt.plot(x, curves_dict_att['RS'], '-ro', label= "DO")
# plt.plot(x, curves_dict_att['SP'], '-go', label= "DO+SP")
# plt.xlabel("Epochs")
# plt.ylabel("Attacker's Regret w.r.t Combined Game")
# plt.title("Regret Curves")
# plt.legend(loc="best")
# plt.show()
import os
from attackgraph import file_op as fp

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


path = os.getcwd() + '/drawing/game_data/' + 'game_BR_selfplay.pkl'
# path = os.getcwd() + '/drawing/game_data/' + 'game_randA.pkl'
# path = os.getcwd() + '/drawing/game_data/' + 'game_sep.pkl'
game = fp.load_pkl(path)

nasheq = game.nasheq
weights_att = []
weights_def = []
labels = []
for epoch in nasheq:
    if epoch == 1:
        continue
    if epoch % 2 == 0:
        labels.append(epoch)
        if epoch > 60:
            weights_def.append(0)
            weights_att.append(0)
            continue
        weights_def.append(nasheq[epoch][0][-1])
        weights_att.append(nasheq[epoch][1][-1])

print(len(weights_att))
print(len(weights_def))
print(len(labels))


# Drawing

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, weights_def, width, label='Defender')
rects2 = ax.bar(x + width/2, weights_att, width, label='Attacker')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Weights')
ax.set_title('Weights on Strategies from Self-play on Random Graph')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#
# autolabel(rects1)
# autolabel(rects2)
#
# fig.tight_layout()

plt.show()
from baselines.common.input import observation_input
from baselines.common.tf_util import adjust_shape
import numpy as np
import tensorflow as tf

import os


# from baselines.deepq.deepq import learn_multi_nets
# from baselines.deepq.deepq import learn
# from .deepq import learn



# ================================================================
# Placeholders
# ================================================================


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplementedError

    def make_feed_dict(self, data):
        """Given data input it to the placeholder(s)."""
        raise NotImplementedError


class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholder."""
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: adjust_shape(self._placeholder, data)}

def is_placeholder(x):
    return type(x) is tf.Tensor and len(x.op.inputs) == 0


class ObservationInput(PlaceholderTfInput):
    def __init__(self, observation_space, name=None):
        """Creates an input placeholder tailored to a specific observation space

        Parameters
        ----------

        observation_space:
                observation space of the environment. Should be one of the gym.spaces types
        name: str
                tensorflow name of the underlying placeholder
        """
        inpt, self.processed_inpt = observation_input(observation_space, name=name)
        super().__init__(inpt)

    def get(self):
        return self.processed_inpt

# Only work for the attacker.
def mask_generator_att(env, obses):
    batch_size = np.shape(obses)[0]
    num_nodes = env.G.number_of_nodes()
    mask = []
    # G_cur = copy.deepcopy(env.G_reserved)
    G_cur = env.G_mask

    for i in np.arange(batch_size):
        state = obses[i][:num_nodes]
        for j in G_cur.nodes:
            G_cur.nodes[j]['state'] = state[j-1]

        _mask = env.attacker.get_att_canAttack_mask(G_cur)

        mask.append(_mask)

    return np.array(mask, dtype=np.float32)



















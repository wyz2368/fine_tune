import random
import numpy as np
from attackgraph import sample_strategy as ss

class Defender(object):

    def __init__(self, G):
        self.num_nodes = G.number_of_nodes()
        self.observation = [0]*self.num_nodes
        self.history = 3
        self.prev_obs = [0]*self.num_nodes*(self.history - 1)

        self.defact = set()
        self.prev_defact = [set()]*self.history

        self.rand_limit = 4


    #TODO: nn should input mask!!!!!!! ALL zeros.
    def def_greedy_action_builder(self, G, timeleft):
        # Everytime sample a strategy from a mixed strategy and assign to self.nn_def.
        self.defact.clear()
        isDup = False
        mask = np.zeros(shape=(1, self.num_nodes+1), dtype=np.float32)

        action_space = self.get_def_actionspace(G)

        while not isDup:
            def_input = self.def_obs_constructor(G, timeleft)
            x = self.nn_def(def_input[None], mask, 0)[0] #corrensponding to baselines
            if not isinstance(x, np.int64):
                raise ValueError("The chosen action is not a numpy 64 integer.")
            action = action_space[x] #TODO: make sure whether x starting from 0 not 1.
            if action == 'pass':
                break
            isDup = (action in self.defact)
            if not isDup:
                self.defact.add(action)

    def def_greedy_action_builder_single(self, G, timeleft, nn_def):
        self.defact.clear()
        isDup = False
        mask = np.zeros(shape=(1, self.num_nodes+1), dtype=np.float32)
        action_space = self.get_def_actionspace(G)

        while not isDup:
            def_input = self.def_obs_constructor(G, timeleft)
            x = nn_def(def_input[None], mask, 0)[0] #corrensponding to baselines
            if not isinstance(x, np.int64):
                raise ValueError("The chosen action is not an integer.")
            action = action_space[x] # x starting from 0.
            if action == 'pass':
                break
            isDup = (action in self.defact)
            if not isDup:
                self.defact.add(action)

    #TODO: Since initialize with zeros, clean these funcs.
    def def_obs_constructor(self, G, timeleft):
        wasdef = self.get_def_wasDefended(G)
        indef = self.get_def_inDefenseSet(G)
        def_input = self.prev_obs + self.observation + wasdef + indef + [timeleft]
        return np.array(def_input, dtype=np.float32)

    # TODO: Can we do matrix operation to get this vector?
    def get_def_wasDefended(self, G):
        wasdef = []
        #old defact is added first.
        for obs in self.prev_defact:
            for node in G.nodes:
                if node in obs:
                    wasdef.append(1)
                else:
                    wasdef.append(0)
        return wasdef

    def get_def_inDefenseSet(self, G):
        indef = []
        for node in G.nodes:
            if node in self.defact:
                indef.append(1)
            else:
                indef.append(0)
        return indef

    def get_def_hadAlert(self, G):
        alert = []
        for node in G.nodes:
            if G.nodes[node]['state'] == 1:
                if random.uniform(0, 1) <= G.nodes[node]['posActiveProb']:
                    alert.append(1)
                else:
                    alert.append(0)
            elif G.nodes[node]['state'] == 0:
                if random.uniform(0, 1) <= G.nodes[node]['posInactiveProb']:
                    alert.append(1)
                else:
                    alert.append(0)
            else:
                raise ValueError("node state is abnormal.")
        return alert

    def get_def_actionspace(self, G):
        actionspace = list(G.nodes) + ['pass']
        return actionspace

    def uniform_strategy(self, G):
        return set(sorted(random.sample(list(G.nodes), self.rand_limit)))

    def cut_prev_obs(self):
        if len(self.prev_obs)/self.num_nodes > self.history - 1:
            self.prev_obs = self.prev_obs[(- self.history + 1)*self.num_nodes:]

    def cut_prev_defact(self):
        if len(self.prev_defact) > self.history:
            self.prev_defact = self.prev_defact[-self.history:]

    def save_defact2prev(self):
        self.prev_defact.append(self.defact.copy())
        self.cut_prev_defact()

    def update_obs(self, obs):
        #TODO: check obs update for the first time step
        self.prev_obs += self.observation #TODO: prev_obs is a list. Do not append.
        self.cut_prev_obs()
        self.observation = obs
    # TODO: in env, feed noisy obs to the observation

    def update_history(self, history):
        self.history = history

    def reset_def(self):
        self.observation = [0]*self.num_nodes
        self.prev_obs = [0] * self.num_nodes * (self.history - 1)
        self.defact.clear()
        self.prev_defact = [set()] * self.history


    def set_current_strategy(self,strategy):
        self.nn_def = strategy

    #TODO: call this while creating env
    def set_env_belong_to(self,env):
        self.myenv = env

    def set_mix_strategy(self,mix):
        self.mix_str = mix

    #TODO: every time updating game, call this.
    def set_str_set(self,set):
        self.str_set = set

    # TODO: call this once one episode is done.
    def sample_and_set_str(self, str_dict = None):
        nn, position = ss.sample_strategy_from_mixed(env=self.myenv, str_set=self.str_set, mix_str=self.mix_str, identity=0, str_dict=str_dict)
        # print(position)
        self.set_current_strategy(nn)
        return position

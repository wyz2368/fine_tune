from attackgraph.parallel_sim import parallel_sim
import numpy as np
from baselines.deepq.load_action import load_action, load_action_class
from attackgraph import file_op as fp
import os
from attackgraph.simulation import series_sim
# from attackgraph.sim_MPI import do_MPI_sim

# def sim_and_modifiy_Series(MPI_flag=False):
def sim_and_modifiy_Series():
    #TODO: make sure this is correct
    print('Begin simulation and modify payoff matrix.')
    path = os.getcwd() + '/data/game.pkl'
    game = fp.load_pkl(path)

    env = game.env
    num_episodes = game.num_episodes

    #TODO: add str first and then calculate payoff
    old_dim, old_dim1 = game.dim_payoff_def()
    new_dim, new_dim1 = game.num_str()
    if old_dim != old_dim1 or new_dim != new_dim1:
        raise ValueError("Payoff dimension does not match.")

    def_str_list = game.def_str
    att_str_list = game.att_str

    position_col_list = []
    position_row_list = []
    for i in range(new_dim-1):
        position_col_list.append((i,new_dim-1))
    for j in range(new_dim):
        position_row_list.append((new_dim-1,j))

    att_col = []
    att_row = []
    def_col = []
    def_row = []
    #TODO: check the path is correct
    for pos in position_col_list:
        idx_def, idx_att = pos
        # if MPI_flag:
        #     aReward, dReward = do_MPI_sim(att_str_list[idx_att], def_str_list[idx_def])
        # else:
        aReward, dReward = series_sim(env, game, att_str_list[idx_att], def_str_list[idx_def], num_episodes)
        att_col.append(aReward)
        def_col.append(dReward)

    for pos in position_row_list:
        idx_def, idx_att = pos
        # if MPI_flag:
        #     aReward, dReward = do_MPI_sim(att_str_list[idx_att], def_str_list[idx_def])
        # else:
        aReward, dReward = series_sim(env, game, att_str_list[idx_att], def_str_list[idx_def], num_episodes)
        att_row.append(aReward)
        def_row.append(dReward)

    game.add_col_att(np.reshape(np.round(np.array(att_col),2),newshape=(len(att_col),1)))
    game.add_col_def(np.reshape(np.round(np.array(def_col),2), newshape=(len(att_col), 1)))
    game.add_row_att(np.round(np.array(att_row),2)[None])
    game.add_row_def(np.round(np.array(def_row),2)[None])

    fp.save_pkl(game, path = path)
    print("Done simulation and modify payoff matrix.")



# def sim_and_modifiy_Series_with_game(game, MPI_flag=False):
def sim_and_modifiy_Series_with_game(game):
    #TODO: make sure this is correct

    print('Begin simulation and modify payoff matrix.')

    env = game.env
    num_episodes = game.num_episodes

    #TODO: add str first and then calculate payoff
    old_dim, old_dim1 = game.dim_payoff_def()
    new_dim, new_dim1 = game.num_str()
    if old_dim != old_dim1 or new_dim != new_dim1:
        raise ValueError("Payoff dimension does not match.")

    def_str_list = game.def_str
    att_str_list = game.att_str

    position_col_list = []
    position_row_list = []
    for i in range(new_dim-1):
        position_col_list.append((i,new_dim-1))
    for j in range(new_dim):
        position_row_list.append((new_dim-1,j))


    # num_tasks = 2 * new_dim - 1

    att_col = []
    att_row = []
    def_col = []
    def_row = []
    #TODO: check the path is correct
    for pos in position_col_list:
        idx_def, idx_att = pos
        # if MPI_flag:
        #     aReward, dReward = do_MPI_sim(att_str_list[idx_att], def_str_list[idx_def])
        # else:
        aReward, dReward = series_sim(env, game, att_str_list[idx_att], def_str_list[idx_def], num_episodes)
        att_col.append(aReward)
        def_col.append(dReward)

    for pos in position_row_list:
        idx_def, idx_att = pos
        # if MPI_flag:
        #     aReward, dReward = do_MPI_sim(att_str_list[idx_att], def_str_list[idx_def])
        # else:
        aReward, dReward = series_sim(env, game, att_str_list[idx_att], def_str_list[idx_def], num_episodes)
        att_row.append(aReward)
        def_row.append(dReward)

    game.add_col_att(np.reshape(np.round(np.array(att_col), 2), newshape=(len(att_col), 1)))
    game.add_col_def(np.reshape(np.round(np.array(def_col), 2), newshape=(len(att_col), 1)))
    game.add_row_att(np.round(np.array(att_row), 2)[None])
    game.add_row_def(np.round(np.array(def_row), 2)[None])

    # fp.save_pkl(game, path = path)
    print("Done simulation and modify payoff matrix.")
    return game
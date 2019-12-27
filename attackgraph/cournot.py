import numpy as np
from attackgraph.gambit_analysis import do_gambit_analysis

cost = 0.5
ub_p = 10
discount = 0.5
bankrupt_threshold1 = -3
bankrupt_threshold2 = -3
bankrupt_penalty = -100
lb_q = 0
ub_q = 11
step_size = 0.5

mono_q = 9.5/2
mono_util = (9.5/2)**2

def bankrpt(u1, u2):
    if u1 > bankrupt_threshold1 and u2 < bankrupt_threshold2:
        u1 += discount * mono_util
        u2 += bankrupt_penalty
    elif u1 < bankrupt_threshold1 and u2 > bankrupt_threshold2:
        u1 += bankrupt_penalty
        u2 += discount * mono_util
    else:
        u1 += bankrupt_penalty
        u2 += bankrupt_penalty

    return u1, u2


def utility(q1, q2):
    if q1 + q2 > ub_p:
        u1 = -cost * q1
        u2 = -cost * q2
        u1, u2 = bankrpt(u1, u2)
    else:
        p = ub_p - q1 - q2
        u1 = q1 * (p - cost)
        u2 = q2 * (p - cost)
    return u1, u2


def BR(nash_idx, nash, p1_payoff):
    dim = len(np.arange(lb_q, ub_q, step_size))
    act = np.zeros(dim)
    for idx, ne in zip(nash_idx, nash):
        act[int(idx)] = ne
    util_vect = np.sum(p1_payoff * act, axis=1)
    util_vect = np.reshape(util_vect, newshape=(len(util_vect),))
    x = np.argmax(util_vect)
    return x/2

def beneficial_dev(nash_idx, nash, p1_payoff):
    dim = len(np.arange(lb_q, ub_q, step_size))
    act = np.zeros(dim)
    for idx, ne in zip(nash_idx, nash):
        act[int(idx)] = ne
    util_vect = np.sum(p1_payoff * act, axis=1)
    util_vect = np.reshape(util_vect, newshape=(len(util_vect),))

    x = np.argsort(util_vect)[-2]
    return x / 2


def rand(str):
    all_acts = np.arange(lb_q, ub_q, step_size)
    diff = np.setdiff1d(all_acts, str)
    x = np.random.choice(diff)
    return x


def create_payoff_matrix():
    dim = len(np.arange(lb_q, ub_q, step_size))
    p1_payoff = np.zeros((dim, dim))
    p2_payoff = np.zeros((dim, dim))

    i = 0
    for q1 in np.arange(lb_q, ub_q, step_size):
        j = 0
        for q2 in np.arange(lb_q, ub_q, step_size):
            u1, u2 = utility(q1, q2)
            p1_payoff[i, j] = u1
            p2_payoff[i, j] = u2
            j += 1

        i += 1
    # print(p1_payoff)
    return p1_payoff, p2_payoff

def extract_submatrix(idx_x, idx_y, matrix):
    submatrix = np.zeros((len(idx_x), len(idx_y)))
    for i, idx in enumerate(idx_x):
        for j, idy in enumerate(idx_y):
            submatrix[i,j] = matrix[int(idx), int(idy)]

    return submatrix

def regret(nash_1, nash_2, str_p1, str_p2, subgame_u1, subgame_u2, p1_payoff, p2_payoff):
    nash_1 = np.reshape(nash_1, newshape=(len(nash_1),1))
    ne_u1 = np.sum(nash_1 * subgame_u1 * nash_2)
    ne_u2 = np.sum(nash_1 * subgame_u2 * nash_2)

    dim, _ = np.shape(p1_payoff)

    ne_1 = np.zeros(dim)
    ne_2 = np.zeros(dim)

    for i, value in zip(str_p1, nash_1):
        ne_1[int(i*2)] = value
    ne_1 = np.reshape(ne_1, newshape=(len(ne_1), 1))

    for i, value in zip(str_p2, nash_2):
        ne_2[int(i*2)] = value


    max_u1 = np.max(np.sum(p1_payoff * ne_2, axis=1))
    max_u2 = np.max(np.sum(ne_1 * p2_payoff, axis=0))

    regret_p1 = np.maximum(max_u1 - ne_u1, 0)
    regret_p2 = np.maximum(max_u2 - ne_u2, 0)

    return np.maximum(regret_p1, regret_p2)

def NashConv(nash_1, nash_2, str_p1, str_p2, subgame_u1, subgame_u2, p1_payoff, p2_payoff):
    nash_1 = np.reshape(nash_1, newshape=(len(nash_1),1))
    ne_u1 = np.sum(nash_1 * subgame_u1 * nash_2)
    ne_u2 = np.sum(nash_1 * subgame_u2 * nash_2)

    dim, _ = np.shape(p1_payoff)

    ne_1 = np.zeros(dim)
    ne_2 = np.zeros(dim)

    for i, value in zip(str_p1, nash_1):
        ne_1[int(i*2)] = value
    ne_1 = np.reshape(ne_1, newshape=(len(ne_1), 1))

    for i, value in zip(str_p2, nash_2):
        ne_2[int(i*2)] = value


    max_u1 = np.max(np.sum(p1_payoff * ne_2, axis=1))
    max_u2 = np.max(np.sum(ne_1 * p2_payoff, axis=0))

    regret_p1 = np.maximum(max_u1 - ne_u1, 0)
    regret_p2 = np.maximum(max_u2 - ne_u2, 0)

    return regret_p1 + regret_p2

def run(p1_payoff, p2_payoff):
    np.random.seed(0)
    regret_list = []
    str_p1 = []
    str_p2 = []
    epoch = 0
    x1, x2 = 0, 0
    str_p1.append(x1)
    str_p2.append(x2)
    subgame_u1 = extract_submatrix(np.array(str_p1) * 2, np.array(str_p2) * 2, p1_payoff)
    subgame_u2 = extract_submatrix(np.array(str_p1) * 2, np.array(str_p2) * 2, p2_payoff)
    is_terminal = True
    switch = False
    while is_terminal:
        epoch += 1
        nelist = do_gambit_analysis(subgame_u1, subgame_u2, return_list=True)
        # nash_2, nash_1 = do_gambit_analysis(subgame_u1, subgame_u2, maxent=False, minent=True)
        nash_2, nash_1 = do_gambit_analysis(subgame_u1, subgame_u2, maxent=True, minent=False)
        regret_list.append(regret(nash_1, nash_2, np.array(str_p1), np.array(str_p2), subgame_u1, subgame_u2, p1_payoff, p2_payoff))

        # DO solver
        if switch:
            x1 = BR(np.array(str_p2) * 2, nash_2, p1_payoff)
            x2 = BR(np.array(str_p1) * 2, nash_1, p1_payoff)

        # Beneficial Deviation
        if not switch:
            x1 = beneficial_dev(np.array(str_p2) * 2, nash_2, p1_payoff)
            x2 = beneficial_dev(np.array(str_p1) * 2, nash_1, p1_payoff)

        # random
        # x1 = rand(np.array(str_p1))
        # x2 = rand(np.array(str_p2))

        if epoch == 10:
            switch = True

        str_p1.append(x1)
        str_p2.append(x2)

        print("--------------------------------")
        print("Current Epoch is ", epoch)
        print("ne_list:", nelist)
        print("Current NE is ", nash_1, nash_2)
        print("x1:", str_p1)
        print("x2:", str_p2)


        # if x1 not in str_p1:
        #     str_p1.append(x1)
        # if x2 not in str_p2:
        #     str_p2.append(x2)



        subgame_u1 = extract_submatrix(np.array(str_p1) * 2, np.array(str_p2) * 2, p1_payoff)
        subgame_u2 = extract_submatrix(np.array(str_p1) * 2, np.array(str_p2) * 2, p2_payoff)

        if epoch == 20:
            is_terminal = False
            print(regret_list)
            print("x1:", str_p1)
            print("x2:", str_p2)

p1_payoff, p2_payoff = create_payoff_matrix()
run(p1_payoff, p2_payoff)



















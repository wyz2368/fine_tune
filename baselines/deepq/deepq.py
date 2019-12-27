import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
# from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
# from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

from baselines.deepq.utils import mask_generator_att


from attackgraph import file_op as fp
from attackgraph import json_op as jp
from baselines.common import models

DIR = os.getcwd() + '/'


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        # if path is None:
        #     path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path, scope):
        save_variables(path, scope=scope)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)

def load_all_policies(env, str_set, opp_identity):
    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    if opp_identity == 0: # pick a defender's strategy
        path = DIR + 'defender_strategies/'
    elif opp_identity == 1:
        path = DIR + 'attacker_strategies/'
    else:
        raise ValueError("identity is neither 0 or 1!")

    str_dict = {}

    flag = env.training_flag
    env.set_training_flag(opp_identity)

    count = 1
    for picked_str in str_set:
        if count == 1 and 'epoch1.pkl' in picked_str:
            str_dict[picked_str] = fp.load_pkl(path + picked_str)
            count += 1
            continue
        str_dict[picked_str] = learn(
                        env,
                        network=models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                        total_timesteps=0,
                        load_path= path + picked_str,
                        scope = picked_str + '/'
                    )

    env.set_training_flag(flag)
    return str_dict

def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          scope='deepq',
          **network_kwargs
            ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model


    sess = get_session()
    # tf.reset_default_graph()
    # sess = tf.Session()
    # sess.__enter__()

    set_global_seeds(seed)

    training_flag = env.training_flag
    if training_flag == 0:
        num_actions = env.act_dim_def()
    elif training_flag == 1:
        num_actions = env.act_dim_att()
    else:
        raise ValueError("Training flag error!")

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph


    # TODO: make everything smooth.
    if training_flag == 0:
        observation_space_shape = [env.obs_dim_def()]
    elif training_flag == 1:
        observation_space_shape = [env.obs_dim_att()]
    else:
        raise ValueError("Training flag error!")

    def make_obs_ph(name):
        return U.BatchInput(observation_space_shape, name=name)

    # #TODO: Modification to be done

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=num_actions,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        scope=scope

    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': num_actions,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    # obs = env.reset_everything_with_return() #TODO: check type and shape of obs. should be [0.2, 0.4, 0.4] numpy
    reset = True

    if total_timesteps != 0:
        if training_flag == 0:  # defender is training
            env.attacker.sample_and_set_str()
        elif training_flag == 1:  # attacker is training
            env.defender.sample_and_set_str()
        else:
            raise ValueError("Training flag is wrong")

    obs = env.reset_everything_with_return()

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file, scope=scope)
            # logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path, scope=scope)
            # logger.log('Loaded model from {}'.format(load_path))


        for t in range(total_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            #TODO: add mask to act
            if training_flag == 0: # the defender is training
                mask_t = np.zeros(shape=(1, num_actions), dtype=np.float32)
            elif training_flag == 1:
                # mask_t should be a function of obs
                mask_t = mask_generator_att(env, np.array(obs)[None]) # TODO: add one dim
            else:
                raise ValueError("training flag error!")

            action = act(np.array(obs)[None], mask_t, training_flag, update_eps=update_eps, **kwargs)[0]
            #TODO: Modification done.
            env_action = action
            reset = False
            new_obs, rew, done = env.step(env_action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset_everything_with_return()
                episode_rewards.append(0.0)
                reset = True
                if total_timesteps != 0:
                    if training_flag == 0:  # defender is training
                        env.attacker.sample_and_set_str()
                    elif training_flag == 1:  # attacker is training
                        env.defender.sample_and_set_str()
                    else:
                        raise ValueError("Training flag is wrong")


            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                # TODO: add mask_tp1 to train
                # mask_t = mask_generator_att(env,obses_t)
                if training_flag == 0:
                    mask_tp1 = np.zeros(shape=(batch_size, num_actions), dtype=np.float32)
                elif training_flag == 1:
                    mask_tp1 = mask_generator_att(env,obses_tp1) #TODO: check if we need mask for t here.
                else:
                    raise ValueError("training flag error!")
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights, mask_tp1, training_flag)
                # TODO: Modification Done
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            # if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                # logger.record_tabular("steps", t)
                # logger.record_tabular("episodes", num_episodes)
                # logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                # logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                # logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    # if print_freq is not None:
                        # logger.log("Saving model due to mean reward increase: {} -> {}".format(
                        #            saved_mean_reward, mean_100ep_reward))
                    save_variables(model_file, scope=scope)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            # if print_freq is not None:
                # logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act


def learn_multi_nets(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=None,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          scope='deepq',
          **network_kwargs
            ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model


    # sess = get_session()
    # tf.reset_default_graph()

    cur_graph = tf.Graph()
    with cur_graph.as_default():
        sess = tf.Session(graph=cur_graph)
        sess.__enter__()

        set_global_seeds(seed)

        training_flag = env.training_flag

        if training_flag == 0:
            num_actions = env.act_dim_def()
        elif training_flag == 1:
            num_actions = env.act_dim_att()
        else:
            raise ValueError("Training flag error!")


        q_func = build_q_func(network, **network_kwargs)

        # capture the shape outside the closure so that the env object is not serialized
        # by cloudpickle when serializing make_obs_ph


        # TODO: make everything smooth.
        if training_flag == 0:
            observation_space_shape = [env.obs_dim_def()]
        elif training_flag == 1:
            observation_space_shape = [env.obs_dim_att()]
        else:
            raise ValueError("Training flag error!")

        def make_obs_ph(name):
            return U.BatchInput(observation_space_shape, name=name)

        # #TODO: Modification to be done

        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            gamma=gamma,
            grad_norm_clipping=10,
            param_noise=param_noise,
            scope=scope
        )

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': num_actions,
        }

        act = ActWrapper(act, act_params)

        # Create the replay buffer
        if prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = total_timesteps
            beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                           initial_p=prioritized_replay_beta0,
                                           final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(buffer_size)
            beta_schedule = None
        # Create the schedule for exploration starting from 1.
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        saved_mean_reward = None
        # obs = env.reset_everything_with_return() #TODO: check type and shape of obs. should be [0.2, 0.4, 0.4] numpy
        reset = True

        # TODO: Sample from mixed strategy for each episode.
        if total_timesteps != 0:
            if training_flag == 0: # defender is training
                env.attacker.sample_and_set_str()
            elif training_flag == 1: # attacker is training
                env.defender.sample_and_set_str()
            else:
                raise ValueError("Training flag is wrong")

        obs = env.reset_everything_with_return()

        # TODO: Done

        with tempfile.TemporaryDirectory() as td:
            td = checkpoint_path or td

            model_file = os.path.join(td, "model")
            model_saved = False

            if tf.train.latest_checkpoint(td) is not None:
                load_variables(model_file)
                # logger.log('Loaded model from {}'.format(model_file))
                model_saved = True
            elif load_path is not None:
                load_variables(load_path)
                # logger.log('Loaded model from {}'.format(load_path))


            for t in range(total_timesteps):
                if callback is not None:
                    if callback(locals(), globals()):
                        break
                # Take action and update exploration to the newest value
                kwargs = {}
                if not param_noise:
                    update_eps = exploration.value(t)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                #TODO: add mask to act
                if training_flag == 0: # the defender is training
                    mask_t = np.zeros(shape=(1, num_actions), dtype=np.float32)
                elif training_flag == 1:
                    # mask_t should be a function of obs
                    mask_t = mask_generator_att(env, np.array(obs)[None]) #TODO: add one dim
                else:
                    raise ValueError("training flag error!")

                action = act(np.array(obs)[None], mask_t, training_flag, update_eps=update_eps, **kwargs)[0]
                #TODO: Modification done.
                env_action = action
                reset = False
                new_obs, rew, done = env.step(env_action)
                # Store transition in the replay buffer.
                replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                episode_rewards[-1] += rew
                if done:
                    obs = env.reset_everything_with_return()
                    episode_rewards.append(0.0)
                    reset = True
                    if total_timesteps != 0:
                        if training_flag == 0:  # defender is training
                            env.attacker.sample_and_set_str()
                        elif training_flag == 1:  # attacker is training
                            env.defender.sample_and_set_str()
                        else:
                            raise ValueError("Training flag is wrong")

                if t > learning_starts and t % train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if prioritized_replay:
                        experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                        weights, batch_idxes = np.ones_like(rewards), None
                    # TODO: add mask_tp1 to train
                    # mask_t = mask_generator_att(env,obses_t)
                    if training_flag == 0:
                        mask_tp1 = np.zeros(shape=(batch_size, num_actions), dtype=np.float32)
                    elif training_flag == 1:
                        mask_tp1 = mask_generator_att(env,obses_tp1) #TODO: check if we need mask for t here.
                    else:
                        raise ValueError("training flag error!")
                    td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights, mask_tp1, training_flag)
                    # TODO: Modification Done
                    if prioritized_replay:
                        new_priorities = np.abs(td_errors) + prioritized_replay_eps
                        replay_buffer.update_priorities(batch_idxes, new_priorities)

                if t > learning_starts and t % target_network_update_freq == 0:
                    # Update target network periodically.
                    update_target()

                mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
                num_episodes = len(episode_rewards)
                # if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                #     logger.record_tabular("steps", t)
                #     logger.record_tabular("episodes", num_episodes)
                #     logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                #     logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                #     logger.dump_tabular()

                # if (checkpoint_freq is not None and t > learning_starts and
                #         num_episodes > 100 and t % checkpoint_freq == 0):
                #     if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                #         # if print_freq is not None:
                #         #     logger.log("Saving model due to mean reward increase: {} -> {}".format(
                #         #                saved_mean_reward, mean_100ep_reward))
                #         save_variables(model_file, scope=scope)
                #         model_saved = True
                #         saved_mean_reward = mean_100ep_reward
            # if model_saved:
            #     # if print_freq is not None:
            #     #     logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            #     load_variables(model_file)

    return act

class Learner(object):
    def __init__(self, retrain = False, freq=100000):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.retrain = retrain
        self.retrain_freq = freq

    def learn_multi_nets(self,
                         env,
                         network,
                         seed=None,
                         lr=5e-4,
                         total_timesteps=100000,
                         buffer_size=30000,
                         exploration_fraction=0.1,
                         exploration_final_eps=0.02,
                         train_freq=1,
                         batch_size=32,
                         print_freq=100,
                         checkpoint_freq=10000,
                         checkpoint_path=None,
                         learning_starts=1000, # TODO: the should be adjusted when retrained since str_0 should play agaist hado_str.
                         gamma=1.0,
                         target_network_update_freq=500,
                         prioritized_replay=False,
                         prioritized_replay_alpha=0.6,
                         prioritized_replay_beta0=0.4,
                         prioritized_replay_beta_iters=None,
                         prioritized_replay_eps=1e-6,
                         param_noise=False,
                         callback=None,
                         load_path=None,
                         scope='deepq',
                         epoch=-1,
                         fine_tuning=False,
                         **network_kwargs
                         ):
        """Train a deepq model.

        Parameters
        -------
        env: gym.Env
            environment to train on
        network: string or a function
            neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
            (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
            will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
        seed: int or None
            prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
        lr: float
            learning rate for adam optimizer
        total_timesteps: int
            number of env steps to optimizer for
        buffer_size: int
            size of the replay buffer
        exploration_fraction: float
            fraction of entire training period over which the exploration rate is annealed
        exploration_final_eps: float
            final value of random action probability
        train_freq: int
            update the model every `train_freq` steps.
            set to None to disable printing
        batch_size: int
            size of a batched sampled from replay buffer for training
        print_freq: int
            how often to print out training progress
            set to None to disable printing
        checkpoint_freq: int
            how often to save the model. This is so that the best version is restored
            at the end of the training. If you do not wish to restore the best version at
            the end of the training set this variable to None.
        learning_starts: int
            how many steps of the model to collect transitions for before learning starts
        gamma: float
            discount factor
        target_network_update_freq: int
            update the target network every `target_network_update_freq` steps.
        prioritized_replay: True
            if True prioritized replay buffer will be used.
        prioritized_replay_alpha: float
            alpha parameter for prioritized replay buffer
        prioritized_replay_beta0: float
            initial value of beta for prioritized replay buffer
        prioritized_replay_beta_iters: int
            number of iterations over which beta will be annealed from initial value
            to 1.0. If set to None equals to total_timesteps.
        prioritized_replay_eps: float
            epsilon to add to the TD errors when updating priorities.
        param_noise: bool
            whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
        callback: (locals, globals) -> None
            function called at every steps with state of the algorithm.
            If callback returns true training stops.
        load_path: str
            path to load the model from. (default: None)
        **network_kwargs
            additional keyword arguments to pass to the network builder.

        Returns
        -------
        act: ActWrapper
            Wrapper over act function. Adds ability to save it and load it.
            See header of baselines/deepq/categorical.py for details on the act function.
        """
        # Create all the functions necessary to train the model

        # sess = get_session()
        # tf.reset_default_graph()

        with self.graph.as_default():
            with self.sess.as_default():

                set_global_seeds(seed)

                training_flag = env.training_flag
                if training_flag == 0:
                    num_actions = env.act_dim_def()
                elif training_flag == 1:
                    num_actions = env.act_dim_att()
                else:
                    raise ValueError("Training flag error!")

                q_func = build_q_func(network, **network_kwargs)

                # capture the shape outside the closure so that the env object is not serialized
                # by cloudpickle when serializing make_obs_ph

                # TODO: make everything smooth.
                if training_flag == 0:
                    observation_space_shape = [env.obs_dim_def()]
                    retrain_path = os.getcwd() + '/retrain_def/'
                    retrain_name = 'def_str_retrain'
                elif training_flag == 1:
                    observation_space_shape = [env.obs_dim_att()]
                    retrain_path = os.getcwd() + '/retrain_att/'
                    retrain_name = 'att_str_retrain'
                else:
                    raise ValueError("Training flag error!")


                def make_obs_ph(name):
                    return U.BatchInput(observation_space_shape, name=name)

                # #TODO: Modification to be done

                act, train, update_target, debug = deepq.build_train(
                    make_obs_ph=make_obs_ph,
                    q_func=q_func,
                    num_actions=num_actions,
                    optimizer=tf.train.AdamOptimizer(learning_rate=lr),
                    gamma=gamma,
                    grad_norm_clipping=10,
                    param_noise=param_noise,
                    scope=scope
                )

                act_params = {
                    'make_obs_ph': make_obs_ph,
                    'q_func': q_func,
                    'num_actions': num_actions,
                }

                act = ActWrapper(act, act_params)

                # Create the replay buffer
                if prioritized_replay:
                    replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
                    if prioritized_replay_beta_iters is None:
                        prioritized_replay_beta_iters = total_timesteps
                    beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                   initial_p=prioritized_replay_beta0,
                                                   final_p=1.0)
                else:
                    replay_buffer = ReplayBuffer(buffer_size)
                    beta_schedule = None
                # Create the schedule for exploration starting from 1.
                exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                             initial_p=1.0,
                                             final_p=exploration_final_eps)

                # Initialize the parameters and copy them to the target network.
                U.initialize()
                update_target()

                mean_rew_list = []
                retrain_episode_rewards = []
                episode_rewards = [0.0]
                temp_buffer = []
                if training_flag == 0:
                    harmfulness = np.zeros(len(env.attacker.str_set))
                    moving_average = np.zeros(len(env.attacker.str_set))
                elif training_flag == 1:
                    harmfulness = np.zeros(len(env.defender.str_set))
                    moving_average = np.zeros(len(env.defender.str_set))
                else:
                    raise ValueError("Training flag error!")

                saved_mean_reward = None
                # obs = env.reset_everything_with_return()  # TODO: check type and shape of obs. should be [0.2, 0.4, 0.4] numpy
                reset = True

                one_hot_att = False
                one_hot_def = False

                add_first_rew = True

                if fine_tuning and load_path is not None:
                    load_variables(load_path, sess=self.sess)

                if total_timesteps != 0:
                    if training_flag == 0:  # defender is training
                        if len(np.where(env.attacker.mix_str>0.95)[0]) == 1: # if pure strategy
                            position = env.attacker.sample_and_set_str()
                            one_hot_att = True
                        else: # if mixed strategy
                            str_dict = load_all_policies(env, env.attacker.str_set, opp_identity=1)
                            position = env.attacker.sample_and_set_str(str_dict=str_dict)
                    elif training_flag == 1:  # attacker is training
                        if len(np.where(env.defender.mix_str>0.95)[0]) == 1: # if pure strategy
                            position = env.defender.sample_and_set_str()
                            one_hot_def = True
                        else: # if mixed strategy
                            str_dict = load_all_policies(env, env.defender.str_set, opp_identity=0)
                            position = env.defender.sample_and_set_str(str_dict=str_dict)
                    else:
                        raise ValueError("Training flag is wrong")

                obs = env.reset_everything_with_return()


                with tempfile.TemporaryDirectory() as td:
                    td = checkpoint_path or td

                    model_file = os.path.join(td, "model")
                    model_saved = False

                    if tf.train.latest_checkpoint(td) is not None and not self.retrain:
                        # load_variables(model_file, sess=self.sess)
                        # logger.log('Loaded model from {}'.format(model_file))
                        # model_saved = True
                        a = 0
                    elif load_path is not None and not fine_tuning:
                        load_variables(load_path, sess=self.sess)
                        # logger.log('Loaded model from {}'.format(load_path))

                    for t in range(total_timesteps):
                        if callback is not None:
                            if callback(locals(), globals()):
                                break
                        # Take action and update exploration to the newest value
                        kwargs = {}
                        if not param_noise:
                            update_eps = exploration.value(t)
                            update_param_noise_threshold = 0.
                        else:
                            update_eps = 0.
                            # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                            # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                            # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                            # for detailed explanation.
                            update_param_noise_threshold = -np.log(
                                1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                            kwargs['reset'] = reset
                            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                            kwargs['update_param_noise_scale'] = True
                        # TODO: add mask to act
                        if training_flag == 0:  # the defender is training
                            mask_t = np.zeros(shape=(1, num_actions), dtype=np.float32)
                        elif training_flag == 1:
                            # mask_t should be a function of obs
                            mask_t = mask_generator_att(env, np.array(obs)[None]) #TODO: add one dim to obs
                        else:
                            raise ValueError("training flag error!")


                        action = act(np.array(obs)[None], mask_t, training_flag, update_eps=update_eps, **kwargs)[0]
                        # TODO: Modification done.
                        env_action = action
                        reset = False

                        new_obs, rew, done = env.step(env_action)
                        # print('time', t, 'env_action:', env_action)
                        # Store transition in the replay buffer.
                        # temp_buffer.append((obs, action, rew, new_obs, float(done)))
                        # replay_buffer.add(obs, action, rew, new_obs, float(done))

                        pass_flag = False
                        if training_flag == 0:
                            rewards_shaping = env.rewards()
                            if rewards_shaping['pass_flag']:
                                for transition in temp_buffer:
                                    obs0, action0, rew0, new_obs0, done0 = transition
                                    # print('transtion', rew0)
                                    rew_new = rewards_shaping[str(action0)].v
                                    episode_rewards[-1] += rew_new
                                    replay_buffer.add(obs0, action0, rew_new, new_obs0, done0)
                                    # print('changed:', rew_new, 'action:', action0)
                                temp_buffer = []
                                # print('********')
                                env.reset_reward_shaping()
                                pass_flag = True
                        elif training_flag == 1:
                            rewards_shaping = env.rewards()
                            if rewards_shaping['pass_flag']:
                                for transition in temp_buffer:
                                    obs1, action1, rew1, new_obs1, done1 = transition
                                    # print('transtion', rew1)
                                    rew_new = rewards_shaping[str(action1)].v
                                    episode_rewards[-1] += rew_new
                                    # print('act:', action1, 'rew_new:', rew_new, 'rew1:', rew1)
                                    replay_buffer.add(obs1, action1, rew_new, new_obs1, done1)
                                    # print('changed:', rew_new, 'action:', action1)
                                temp_buffer = []
                                # print('********')
                                env.reset_reward_shaping()
                                pass_flag = True


                        if pass_flag:
                            episode_rewards[-1] += rew
                            replay_buffer.add(obs, action, rew, new_obs, float(done))
                        else:
                            temp_buffer.append((obs, action, rew, new_obs, float(done)))

                        obs = new_obs

                        if done:
                            obs = env.reset_everything_with_return()

                            if t > 0.95 * total_timesteps:
                                moving_average[position] += 1
                                harmfulness[position] += (episode_rewards[-1]-harmfulness[position])/moving_average[position]


                            episode_rewards.append(0.0)
                            reset = True
                            # sample a new strategy from meta-stategy solver.
                            if not one_hot_att and not one_hot_def:
                                if total_timesteps != 0:
                                    if training_flag == 0:  # defender is training
                                        position = env.attacker.sample_and_set_str(str_dict)
                                    elif training_flag == 1:  # attacker is training
                                        position = env.defender.sample_and_set_str(str_dict)
                                    else:
                                        raise ValueError("Training flag is wrong")

                        if t > learning_starts and t % train_freq == 0:
                            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                            if prioritized_replay:
                                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                            else:
                                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                                weights, batch_idxes = np.ones_like(rewards), None
                            # TODO: add mask_tp1 to train
                            # mask_t = mask_generator_att(env,obses_t)
                            if training_flag == 0:
                                mask_tp1 = np.zeros(shape=(batch_size, num_actions), dtype=np.float32)
                            elif training_flag == 1:
                                mask_tp1 = mask_generator_att(env, obses_tp1)  # TODO: check if we need mask for t here.
                            else:
                                raise ValueError("training flag error!")
                            td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights, mask_tp1, training_flag)
                            # TODO: Modification Done
                            if prioritized_replay:
                                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                                replay_buffer.update_priorities(batch_idxes, new_priorities)

                        if t > learning_starts and t % target_network_update_freq == 0:
                            # Update target network periodically.
                            update_target()

                        mean_100ep_reward = round(np.mean(episode_rewards[-251:-1]), 1)
                        mean_rew_list.append(mean_100ep_reward)
                        num_episodes = len(episode_rewards)

                        if (checkpoint_freq is not None and t > learning_starts and
                                num_episodes > 100 and t % checkpoint_freq == 0) and not self.retrain:
                            if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                                # if print_freq is not None:
                                #     logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                #         saved_mean_reward, mean_100ep_reward))
                                save_variables(model_file, sess=self.sess, scope=scope)
                                model_saved = True
                                saved_mean_reward = mean_100ep_reward


                    if model_saved and not self.retrain:
                        # if print_freq is not None:
                        #     logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
                        load_variables(model_file, sess=self.sess, scope=scope)
                        if saved_mean_reward is not None and saved_mean_reward > mean_100ep_reward:
                            BD = saved_mean_reward
                        else:
                            BD = mean_100ep_reward
                    else:
                        BD = None


                    if total_timesteps != 0:
                        if training_flag == 0:
                            path = os.getcwd() + '/learning_curve/def_data' + str(epoch) + '.pkl'
                            fp.save_pkl(mean_rew_list,path)
                        elif training_flag == 1:
                            path = os.getcwd() + '/learning_curve/att_data' + str(epoch) + '.pkl'
                            fp.save_pkl(mean_rew_list, path)
                        else:
                            raise ValueError("Error training flag.")


        if total_timesteps == 0:
            return act

        return act, BD , harmfulness


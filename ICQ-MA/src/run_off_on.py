import datetime
import os
import pprint
from textwrap import fill
import time
import math as mth
import numpy as np
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import h5py
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, Best_experience_Buffer
from components.transforms import OneHot
import datetime


device = th.device("cuda" if th.cuda.is_available() else "cpu")


class ReplayBuffer_t:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.not_done_buf = np.zeros(size, dtype=np.float32)
        self.discounted_returns = np.array([])
        self.early_cut = np.array([])
        self.terminate_states = np.array([])

        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def approx_v(self,state,q,pi):
        M = 20
        with torch.no_grad():
            states = state.repeat(M, 1)
            actions,logp_actions = pi(states)
            q_value_list = q(states, actions)
            # q1_pi = self.ac_targ.q1(states, actions)
            # q2_pi = self.ac_targ.q2(states, actions)
            qs = torch.min(q_value_list)
            # qs = self.ac_targ.q1(states, actions)
            # qs = qs-self.alpha*logp_actions # with entroy
            qs_split = torch.split(qs, state.shape[0])
            qs_mean = torch.mean(torch.stack(qs_split), dim=0)
        return qs_mean


    def calculate_nstep_adv(self,q,pi,state,idx):
        with torch.no_grad():
            v = self.approx_v(state,q,pi)
            dicounted_return = self.calcuate_n_step_return(idx)
            v_cpu = v.detach().cpu().numpy()
            n_adv = dicounted_return - v_cpu

        return n_adv,dicounted_return,v_cpu


    def calcuate_n_step_return(self,ind_start):
        discounted_return = self.discounted_returns[ind_start].reshape(-1)
        terminal_states_idx = self.terminate_states[ind_start]

        early_cut = self.early_cut[terminal_states_idx]
        early_cut_state_idx = np.where(early_cut == 1)
        is_empty = early_cut_state_idx[0].size == 0

        if is_empty:
            return discounted_return

        early_cut_state_idx = terminal_states_idx[early_cut_state_idx]
        terminal_states_unique = np.unique(terminal_states_idx)

        terminal_states = torch.FloatTensor(
            self.buf.next_state[terminal_states_unique]).to(device)

        # *torch.FloatTensor(early_cut)#.to(device)
        v_last = self.approx_v(terminal_states)

        for state_idx, v in zip(terminal_states_unique, v_last):
            v_last_idx = np.where(terminal_states_idx == state_idx)
            # v need to multiple gamma's N
            discounted_return[v_last_idx] += v.item()*self.gamma

        return discounted_return


    def load(self, save_folder, size=-1):
        reward = [];not_done = []
        reward_buffer = np.load(f"{save_folder}_reward.npy")
        not_done_buffer = np.load(f"{save_folder}_not_done.npy")
        
        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.obs_buf[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
        self.act_buf[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
        self.obs2_buf[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
        for r,n_d in zip(reward_buffer,not_done_buffer):
            reward.append(float(r[0]))
            not_done.append(int(n_d[0]))
        self.rew_buf[:self.size] = reward[:self.size]
        self.not_done_buf[:self.size] = not_done[:self.size]
        self.done_buf[:self.size] = 1-self.not_done_buf[:self.size]

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.obs_buf[idxs],
                     next_state=self.obs2_buf[idxs],
                     action=self.act_buf[idxs],
                     reward=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     not_done=self.not_done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}


# all tensor store in the same buffer, 
class replaybuffer_mine:
    def __init__(self) -> None:
        self.current_online_size = 0
        self.size = 0
        self.online_window = 200
        
    def add_online_batch(self,batch):# sample trajectories not tuples
        # c = torch.cat((c, ones), 1)
        onlinesize = len(batch.data.transition_data['actions'])
        if self.current_online_size < self.online_window:
            self.current_online_size += onlinesize
        self.size += onlinesize

        self.actions_h = th.cat((self.actions_h,batch.data.transition_data['actions']),0)
        self.actions_onehot_h = th.cat((self.actions_onehot_h,batch.data.transition_data['actions_onehot']),0)
        self.avail_actions_h = th.cat((self.avail_actions_h,batch.data.transition_data['avail_actions']),0)
        self.filled_h = th.cat((self.filled_h,batch.data.transition_data['filled']),0)
        self.obs_h = th.cat((self.obs_h,batch.data.transition_data['obs']),0)
        self.reward_h = th.cat((self.reward_h,batch.data.transition_data['reward']),0)
        self.state_h = th.cat((self.state_h,batch.data.transition_data['state']),0)
        self.terminated_h = th.cat((self.terminated_h,batch.data.transition_data['terminated']),0)

        return 
        

    def addh5py(self,hdFile_r):
        self.actions_h = th.tensor(hdFile_r.get('actions')).to(device)
        self.actions_onehot_h = th.tensor(hdFile_r.get('actions_onehot')).to(device)
        self.avail_actions_h = th.tensor(hdFile_r.get('avail_actions')).to(device)
        self.filled_h = th.tensor(hdFile_r.get('filled')).to(device)
        self.obs_h = th.tensor(hdFile_r.get('obs')).to(device)
        self.reward_h = th.tensor(hdFile_r.get('reward')).to(device)
        self.state_h = th.tensor(hdFile_r.get('state')).to(device)
        self.terminated_h = th.tensor(hdFile_r.get('terminated')).to(device)
        self.size = len(self.actions_h) # 100

    def random_sample(self,batch_size=32,online=False):
        if online and self.current_online_size > 100:
            sample_number = np.random.choice(range(self.size-self.current_online_size,self.size), batch_size, replace=False)
        else:
            sample_number = np.random.choice(range(0,self.size), batch_size, replace=False)
        
        filled_sample = self.filled_h[sample_number]
        max_ep_t_h = filled_sample.sum(1).max(0)[0]
        filled_sample = filled_sample[:, :max_ep_t_h]
        actions_sample = self.actions_h[sample_number][:, :max_ep_t_h]
        actions_onehot_sample = self.actions_onehot_h[sample_number][:, :max_ep_t_h]
        avail_actions_sample = self.avail_actions_h[sample_number][:, :max_ep_t_h]
        obs_sample = self.obs_h[sample_number][:, :max_ep_t_h]
        reward_sample = self.reward_h[sample_number][:, :max_ep_t_h]
        state_sample = self.state_h[sample_number][:, :max_ep_t_h]
        terminated_sample = self.terminated_h[sample_number][:, :max_ep_t_h]

        batch = {}
        batch['obs'] = obs_sample
        batch['reward'] = reward_sample
        batch['actions'] = actions_sample
        batch['actions_onehot'] = actions_onehot_sample
        batch['avail_actions'] = avail_actions_sample
        batch['filled'] = filled_sample
        batch['state'] = state_sample
        batch['terminated'] = terminated_sample
        batch['batch_size'] = batch_size
        batch['max_seq_length'] = max_ep_t_h
        return batch


def load_hdf5(dataset, replay_buffer):
    replay_buffer._size = dataset['terminals'].shape[0]

    replay_buffer._observations[:replay_buffer._size] = dataset['observations']
    replay_buffer._next_obs[:replay_buffer._size] = dataset['next_observations']
    replay_buffer._actions[:replay_buffer._size] = dataset['actions']
    replay_buffer._rewards[:replay_buffer._size] = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals[:replay_buffer._size] = np.expand_dims(np.squeeze(dataset['terminals']), 1)  
    # replay_buffer._size = dataset['terminals'].shape[0]
    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    logger.setup_sacred(_run)

    run_sequential(args=args, logger=logger)

    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        episode_batch = runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def offline_training(learner,runner,batch,running_log,beta):
    # --------------------- ICQ-MA --------------------------------
    learner.train_critic(batch, best_batch=None, log=running_log, t_env=runner.t_env,beta=beta) # add greedy
    learner.train(batch, runner.t_env, running_log) # add greedy
    


def online_explore(runner,buffer,n_test_runs=2):
    for _ in range(n_test_runs):
        online_batch = runner.explore()
        buffer.add_online_batch(online_batch)

def run_sequential(args, logger):

    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    online_buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    buffer_mine = replaybuffer_mine()

    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()

    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    start_time = time.time()
    last_time = start_time
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))    
    episode_num = 0
    update_num = 0
    
    # --------------------------- hdf5 -------------------------------
    import h5py
    hdFile_r = h5py.File(args.env_args['map_name'] + '.h5', 'r')

    # actions_h = th.tensor(hdFile_r.get('actions')).to(args.device)
    # actions_onehot_h = th.tensor(hdFile_r.get('actions_onehot')).to(args.device)
    # avail_actions_h = th.tensor(hdFile_r.get('avail_actions')).to(args.device)
    # filled_h = th.tensor(hdFile_r.get('filled')).to(args.device)
    # obs_h = th.tensor(hdFile_r.get('obs')).to(args.device)
    # reward_h = th.tensor(hdFile_r.get('reward')).to(args.device)
    # state_h = th.tensor(hdFile_r.get('state')).to(args.device)
    # terminated_h = th.tensor(hdFile_r.get('terminated')).to(args.device)

    buffer_mine.addh5py(hdFile_r)
    max_steps = 1000000
    offline_steps = 300000

    # ----------------------------pre train-------------------------------
    while runner.t_env <= args.t_max:
        # if runner.t_env >= 4000200:
        #     break

        if learner.critic_training_steps >= max_steps:
            break

        th.set_num_threads(8)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
            "q_max_mean": [],
            "q_min_mean": [],
            "q_max_var": [],
            "q_min_var": []
        }
        if learner.critic_training_steps > offline_steps:# 100000
            online_explore(runner,buffer_mine)

        p = np.random.uniform()

        if learner.critic_training_steps > offline_steps and p < 0.5:
            on_batch = buffer_mine.random_sample(batch_size=32,online=True)
            offline_training(learner,runner,on_batch,running_log,1000) # greedy 
        else:
            off_batch = buffer_mine.random_sample(batch_size=32,online=False)
            offline_training(learner,runner,off_batch,running_log,1000) # default 1000

        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (learner.critic_training_steps - last_test_T) / args.test_interval >= 1.0: # args.test_interval
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("critic_training_steps: {} / {}".format(learner.critic_training_steps, max_steps))
            logger.console_logger.info("current_online_size: {} / {}".format(buffer_mine.current_online_size, buffer_mine.size))

            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, learner.critic_training_steps, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = learner.critic_training_steps
            for _ in range(n_test_runs):#2
                runner.run(learner.critic_training_steps,test_mode=True)

        if args.save_model and (learner.critic_training_steps - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = learner.critic_training_steps
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(learner.critic_training_steps))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

        episode += args.batch_size_run

        if (learner.critic_training_steps - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, learner.critic_training_steps)
            logger.print_recent_stats()
            last_log_T = learner.critic_training_steps
        
        episode_num += 1
        update_num += 1
        # runner.t_env += 100


    # ## ------------------interleave training------------------------------
    # on_batch = {}
    
    # ## ------------------online explore--------------------------
    # for _ in range(n_test_runs):
    #     runner.explore(online_buffer)

    # actions_h.extend(online_data)
    # actions_onehot_h.extend(online_data)
    # avail_actions_h.extend(online_data)
    # filled_h.extend(online_data)
    # obs_h.extend(online_data)
    # reward_h.extend(online_data)
    # state_h.extend(online_data)
    # terminated_h.extend(online_data)

    # on_batch.extend(online_data)

    # p = np.random.uniform()

    # if p < 0.5 and 

    # ## sample from on_batch with 0.5
    #     learner.train_critic(off_batch, best_batch=None, log=running_log, t_env=runner.t_env)
    #     learner.train(off_batch, runner.t_env, running_log)


    ## sample from off_batch with 0.5

    ## if on batch update with optimstic 

    ## else update with pessimistic


    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config


def process_batch(batch, args):

    if batch.device != args.device:
        batch.to(args.device)
    return batch
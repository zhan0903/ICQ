# Implicit Constraint Q-Learning

This is a pytorch implementation of ICQ on [Datasets for Deep Data-Driven Reinforcement Learning (D4RL)](https://github.com/rail-berkeley/d4rl) and ICQ-MA on [SMAC](https://github.com/oxwhirl/smac), the corresponding paper of ICQ is [Believe What You See: Implicit Constraint Approach
for Offline Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2106.03400).

## Requirements
**Single-agent:**

Please enter the `ICQ_mu`, `ICQ_softmax`, `ICQ-antmaze_mu` and `ICQ-antmaze_softmax` folders.
- python=3.6.5
- [Datasets for Deep Data-Driven Reinforcement Learning (D4RL)](https://github.com/rail-berkeley/d4rl)
- torch=1.1.0

**Multi-agent:**

Please enter the `ICQ-MA` folder.
Then, set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The `requirements.txt` file can be used to install the necessary packages into a virtual environment (not recommended).

## Quick Start
**Single-agent:**
```shell
$ python3 main.py
```
**Multi-agent:**
```shell
$ python3 src/main.py --config=offpg_smac --env-config=sc2 with env_args.map_name=3s_vs_3z
```
The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

## Citing
If you find this open source release useful, please reference in your paper (it is our honor):
```
@article{yang2021believe,
  title={Believe What You See: Implicit Constraint Approach for Offline Multi-Agent Reinforcement Learning},
  author={Yang, Yiqin and Ma, Xiaoteng and Li, Chenghao and Zheng, Zewu and Zhang, Qiyuan and Huang, Gao and Yang, Jun and Zhao, Qianchuan},
  journal={arXiv preprint arXiv:2106.03400},
  year={2021}
}
```

## Note
+ If you have any questions, please contact me: yangyiqi19@mails.tsinghua.edu.cn. 
+ The implementation is based on [PyMARL](https://github.com/oxwhirl/pymarl), [SMAC](https://github.com/oxwhirl/smac) codebases and [DOP](https://github.com/TonghanWang/DOP) which are open-sourced.


```diff
- Please pay attention to the version of SC2 you are using for your experiments. 
- Performance is *not* always comparable between versions. 
- The results in PYMARL-RLC use SC2.4.10.
```

# Python MARL framework

PyMARL-RLC includes implementations of the following algorithms:
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [**WQMIX**: Weighted Qmix: Expanding Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf)
- [**Qatten**: Qatten: A General Framework for Cooperative Multiagent Reinforcement Learning](https://arxiv.org/abs/2002.03939)
- [**QPLEX**: Qplex: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062)
- [**MASAC**: Actor-Attention-Critic for Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v97/iqbal19a/iqbal19a.pdf)
- [**SHAQ**: SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning](https://arxiv.org/pdf/2105.15013.pdf)
- [**SQDDPG**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [**UPDET**: Updet: Universal Multi-Agent Reinforcement Learning via Policy Decoupling with Transformers](https://arxiv.org/abs/2101.08001)
- [**CDS**: Celebrating Diversity in Shared Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2106.02195)
- [**MIXRTs**: MIXRTs: Toward Interpretable Multi-Agent Reinforcement Learning via Mixing Recurrent Soft Decision Trees](https://arxiv.org/abs/2209.07225)

PyMARL is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

```shell
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m gpu_id=0 t_max=2000000 epsilon_anneal_time=50000 seed=4
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`


All results will be stored in the `Results` folder.


## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.


## Citing PyMARL-RLC 

If you use PyMARL-RLC in your research, please cite the [SMAC paper](https://arxiv.org/abs/1902.04043) and [MIXRTs paper](https://arxiv.org/abs/2209.07225).

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {The StarCraft Multi-Agent Challenge},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}


@article{liu2022mixrts,
  title={MIXRTs: Toward Interpretable Multi-Agent Reinforcement Learning via Mixing Recurrent Soft Decision Trees},
  author={Liu, Zichuan and Zhu, Yuanyang and Wang, Zhi and Chen, Chunlin},
  journal = {CoRR},
  volume = {abs/2209.07225},
  year={2022}
}
```

## License

Code licensed under the Apache License v2.0

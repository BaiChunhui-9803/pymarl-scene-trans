# PyMARL-run.py

以qmix | MarineMicro_MvsM_4为例

```python
--config=qmix --env-config=sc2 with env_args.map_name=MarineMicro_MvsM_4 t_max=2050000
```

### run.py

#### run()

```python
def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)
    print(_log)
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)
```



#### run_sequential()

```python
def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    
    # Set up schemes and groups here
    env_info = runner.get_env_info()
    
    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    # Groups
    groups = {
        "agents": args.n_agents
    }
    # Preprocess of part of data
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    
```

> [!NOTE]
>
> - `preprocess`：预处理数据，平台在`run.py`中为`'actions'`添加了预处理数据`'actions_onehot'`
>
>   关于动作，采用独热编码`actions_onehot`的意义在于：
>
>   - 简化动作表示：将每个动作表示为一个向量，其中只有一个元素为1，其余元素为0
>   - 兼容性：与许多机器学习模型（如神经网络）更兼容，通常期望输入是固定大小的向量
>   - 避免歧义：避免动作之间的数值关系，动作0和动作1在独热编码中是完全不同的向量，而不是数值上的相邻值
>
> - 



```python
def run_sequential(args, logger):
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    
    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    
    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    
```

> [!NOTE]
>
> - mac
> - `runner.setup()`：为注册的`runner`更新计划`scheme`、智能体组数`groups`、预处理数据`preporcess`、多智能体控制器`mac`





```python
def run_sequential(args, logger):

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    
```





```python
def run_sequential(args, logger):
    ***
    
    # start training
	episode = 0
	last_test_T = -args.test_interval - 1
	last_log_T = 0
	model_save_time = 0

	start_time = time.time()
	last_time = start_time

	logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

```
> [!NOTE]
>
> 



```python
def run_sequential(args, logger):
    ***
    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)


```

> [!NOTE]
>
> 如果`runner.t_env` 小于`args.t_max`，即**未达到停止迭代条件**
>
> 1. `runner.run(test_mode=False)`：完整执行一段episode
>
>    执行完毕之后，一些有用的信息：
>
>    - `runner.t_env`更新为`runner.t_env = runner.t_env + t_episode`，记录自程序启动到目前所经过的总步数`t_env`，每执行一次episode后自增该episode所经过的步数`t_episode`
>
> 2. `buffer.insert_episode_batch(episode_batch)`：将刚刚执行过的episode添加到缓冲区`buffer`中





```python
def run_sequential(args, logger):
    ***
    while runner.t_env <= args.t_max:
        ***
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
```

> [!NOTE]
>
> TODO：buffer采样



```python
def run_sequential(args, logger):
    ***
    while runner.t_env <= args.t_max:
        ***
        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)
```

> [!NOTE]
>
> **偶尔执行一段测试运行，偶尔的含义为每间隔`args.test_interval`执行一次测试运行**
>
> `n_test_runs`：测试运行episode数，至少为1；当`runner.batch_size`为1时，`n_test_runs`为`args.test_nepisode`
>
> `runner.t_env - last_test_T`：当前`t_env`与上次测试运行`last_test_T`的间隔
>
> - 如果测试间隔超过所设置间隔`args.test_interval `
>   - 记录和打印日志信息
>   - 更新时间记录`last_test_T = runner.t_env`
>   - 执行测试运行
>   
>   ```python
>   for _ in range(n_test_runs):
>   	runner.run(test_mode=True)
>   ```
>   
>   以`test_mode=True`模式，循环执行 `n_test_runs` 次测试运行


# PyMARL-episode_runner.py

以qmix | MarineMicro_MvsM_4为例

```python
--config=qmix --env-config=sc2 with env_args.map_name=MarineMicro_MvsM_4 t_max=2050000
```

### episode_runner.py

#### class EpisodeRunner

```python
class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000
```

> [!NOTE]
>
> - `self.batch_size = self.args.batch_size_run`：批次大小`batch_size`的值由配置参数`batch_size_run`决定
> - 在`EpisodeRunner`中，`assert self.batch_size == 1`，意味着episode不涉及并行执行

> [!NOTE]
>
> - 根据参数`args.env`注册`self.env`，通常为sc2环境
> - **`self.t`：跟踪当前 episode 中的时间步数`t`**
>   - 从 0 开始，每次环境步进时递增
>   - 每次开始新的 episode 时，self.t 会被重置为 0
> - **`self.t_env`：跟踪整个训练过程中的全局时间步数`t_env`**
>   - 不会在每个 episode 结束时重置，而是持续累加
>   - `self.t_env`大于`args.t_max`时，程序终止

> [!CAUTION]
>
> 跳帧参数 `_step_mul`仅仅影响了决策频率，不会影响 `self.t` 与 `self.t_env` 的计算方式，即两者的递增只与 `step()` 的执行有关，每执行一次 `step()` ，两个计数器自增1



------

```python
class EpisodeRunner:
    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size,
                                 self.episode_limit + 1,
                                 preprocess=preprocess,
                                 device=self.args.device)
        self.mac = mac
```

> [!NOTE]
>
> 一些预设置，该方法在`run.py`中`run_sequential()`的准备阶段，由注册的`EpisodeRunner`实例调用
>
> - `self.new_batch`：一个部分应用函数，已经预先设置了`EpisodeRunner`参数
> - `self.mac`：





------

**在`runners.py`中注册了`EpisodeRunner`与`ParallelRunner`，本节讨论前者**

**`EpisodeRunner.run()`**

```python
class EpisodeRunner:
    def run(self, test_mode=False):
    	self.reset()
        
    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
```

> [!NOTE]
>
> `self.reset()`主要做三件事
>
> 1. 创建一个新的 `EpisodeBatch` 实例 `self.batch`，用于存储当前 episode 的数据。`self.new_batch` 是一个部分应用函数，该部分应用函数已在 `EpisodeRunner.setup()` 中设置
> 2. 执行 `self.env.reset()`，重置环境到初始状态
> 3. 将时间步长 `t` 重置为 0。`t` 用于跟踪当前 episode 中的时间步数，从 0 开始计数。



```python
class EpisodeRunner:
    def run(self, test_mode=False):	
        ***
    	pre_transition_data = {
		    "state": [self.env.get_state()],
		    "avail_actions": [self.env.get_avail_actions()],
		    "obs": [self.env.get_obs()]
		}
```

> [!NOTE]
>
> **获取预处理数据，包括状态、可执行动作与观测结果**
>
> **这些信息被保留为01矩阵信息，HRL-IM/CBS复现可以不使用之**





```python
class EpisodeRunner:
    def run(self, test_mode=False):	
        ***
    	self.batch.update(pre_transition_data, ts=self.t)
        
        # Pass the entire batch of experiences up till now to the agents
        # Receive the actions for each agent at this timestep in a batch of size 1
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
```

> [!NOTE]
>
> 将整个当前批次 `self.batch` 的历史信息传递给 `self.mac` 控制器，根据 `select_actions()` 获取在时间步长 `t_ep` 的执行动作 `actions` 



```python
class EpisodeRunner:
    def run(self, test_mode=False):	
        ***
    	reward, terminated, env_info = self.env.step(actions[0])
        episode_return += reward
```

> [!NOTE]
>
> 调用 `env.step()` 方法，控制器以传入的动作信息 `actions[0]` 步进
>
> 返回回报 `reward`、是否终止 `terminated`、环境信息 `env_info`
>
> 更新当前episode的整体回报 `episode_return`

> [!CAUTION]
>
> 注意 `env.step()` 与 `env._controller.step()` 的区别
>
>  `env.step()` ：`StarCraft2Env` 封装的环境步进接口，传入参数为动作张量 `actions`
>
>  `env._controller.step()` ：更底层的接口，传入参数为步进的步长，不建议使用



```python
class EpisodeRunner:
    def run(self, test_mode=False):	
        ***
        if not test_mode:
        self.t_env += self.t
```

> [!CAUTION]
>
> 注意，`test_mode` 模式下，环境时间步进 `self.t_env` 不会递增


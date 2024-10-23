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
> 1. 创建一个新的 `EpisodeBatch` 实例，用于存储当前 episode 的数据。`self.new_batch` 是一个部分应用函数，该部分应用函数已在`EpisodeRunner.setup()`中设置
> 2. 执行`self.env.reset()`，重置环境到初始状态
> 3. 将时间步长 `t` 重置为 0。`t` 用于跟踪当前 episode 中的时间步数，从 0 开始计数。












# PyMARL-episode_buffer.py

以qmix | MarineMicro_MvsM_4为例

```python
--config=qmix --env-config=sc2 with env_args.map_name=MarineMicro_MvsM_4 t_max=2050000
```





### episode_buffer.py

#### class EpisodeBatch

```python
class EpisodeBatch:
    def __getitem__(self, item):
```

> [!CAUTION]
>
> `EpisodeBatch` 实现了 `__getitem__` 方法，使对象能够使用索引访问其内部数据
>
> 可以理解为`EpisodeBatch` 对象本身是一个列表或字典，可以通过方括号 `[]` 访问内部元素
>
> `EpisodeBatch` 的主要数据封装在了 `self.data.transition_data` 和 `self.data.episode_data` 两个字典中。如此，以下操作是成立的且等价于 `self.data.transition_data["state"]`

```python
self.new_batch = partial(EpisodeBatch, *, *, *)
self.new_batch["state"]
```

------

```python
class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)
```

> [!NOTE]
>
> `scheme`：数据字典，
>
> - `state`：`torch.Size([5000, 61, 68])`
> - `obs`：`torch.Size([5000, 61, 4, 40])`
> - `actions`：`torch.Size([5000, 61, 4, 1])`
> - `avail_actions`：`torch.Size([5000, 61, 4, 10])`
> - `reward`：`torch.Size([5000, 61, 1])`
> - `terminated`：`torch.Size([5000, 61, 1])`
> - `actions_onehot`：`torch.Size([5000, 61, 4, 10])`
> - `filled`：`torch.Size([5000, 61, 1])`
>
> `groups`：我方智能体个数
>
> `batch_size`：执行批次大小
>
> `max_seq_length`：
>
> `data=None`：
>
> `preprocess=None`：

```python
class EpisodeBatch:
	def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
    	***
    	for field_key, field_info in scheme.items():
        	***
        	self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)
        	***
    	***
```

> [!NOTE]
>
> **为`EpisodeBatch`中的`self.data.transition_data`设置初始data**
>
> `field_key`：需要在`transition_data`中添加的键，键的来源为**`episode_buffer.py setup()`**传入的**`scheme`**
>
> `batch_size`：批次大小，5000
>
> `max_seq_length`：sequence最大长度，一个episode的最大长度，`env.user_map.limit` + 1 = 61
>
> `*shape`：该键表示的张量解包后的维度
>
> | batch_size         |                           |
> | ------------------ | ------------------------- |
> | 5000               |                           |
> | **max_seq_length** | calculated by 'limit + 1' |
> | 61                 |                           |
> | **field_key**      | ***shape**                |
> | 'state'            | 68                        |
> | 'obs'              | 4, 40                     |
> | 'actions'          | 4, 1                      |
> | 'avail_actions'    | 4, 10                     |
> | 'reward'           | 1                         |
> | 'terminated'       | 1                         |
> | 'actions_onehot'   | 4, 10                     |
> | 'filled'           | 1                         |

```python
class EpisodeBatch:
    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])
```

> [!TIP]
>
> `update()`会根据传入的数据类型（transition_data 或 episode_data）分别更新不同的目标数据存储位置
>
> **调用实例1**
>
> ```python
> self.update(ep_batch.data.transition_data,
>          slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
>          slice(0, ep_batch.max_seq_length),
>          mark_filled=False)
> ```
>
> data: 要插入到缓冲区中的过渡数据 `ep_batch.data.transition_data`
>
> bs: 切片对象，在缓冲区中插入数据的位置范围 `slice(self.buffer_index, self.buffer_index + ep_batch.batch_size)` 
>
> ts: 切片对象，表示时间步长的范围，用于指定在时间维度上插入数据的位置 `slice(0, ep_batch.max_seq_length)`
>
> mark_filled: 是否标记插入的数据为已填充
>
> **调用实例2**
>
> ```python
> self.update(ep_batch.data.episode_data,
>          slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
> ```
>
> data: 要插入到缓冲区中的批次数据 `ep_batch.data.episode_data`
>
> bs: 切片对象，在缓冲区中插入数据的位置范围 `slice(self.buffer_index, self.buffer_index + ep_batch.batch_size)` 


> [!NOTE]
>
> **更新缓冲区中的数据，由`ReplayBuffer.insert_episode_batch()`调用**
>
> 1. 解析切片
>
> ```python
> slices = self._parse_slices((bs, ts))
> ```
>
>    - 将传入的切片解析为实际的切片对象
>      -  `bs` 批次切片
>      -  `ts` 时间步长切片
>
>
> ------
>
> 2. 遍历数据字典
>
> ```python
> for k, v in data.items():
> ```
>
>    - 遍历传入的数据字典 data，其中 k 是数据的键，v 是对应的值
>
> ------
>
> 3. 确定目标数据存储位置
> ```python
> if k in self.data.transition_data
> ```
>
>
>    - 键 k 在 `transition_data` 中，表示这是一个过渡数据
>
>      ```python
>      target = self.data.transition_data
>      ```
>
>      将目标数据存储位置设为 transition_data
>
>      ```python
>      if mark_filled:
>          target["filled"][slices] = 1
>          mark_filled = False
>      ```
>
>      如果 mark_filled 为真，标记数据为已填充，将对应切片位置的 filled 标记为 1
>
>      ```python
>      _slices = slices
>      ```
>
>      将切片设为 `slices`
>
> ```python
> elif k in self.data.episode_data:
> ```
>
>
>    - 键 k 在 `episode_data` 中，表示这是一个批次数据
>
>      ```python
>      target = self.data.episode_data
>      ```
>
>      将目标数据存储位置设为 episode_data
>
>      ```python
>      _slices = slices[0]
>      ```
>
>      将切片设为 `slices[0]`，因为批次数据不需要时间步长切片
>
> ```python
> else:
>     raise KeyError("{} not found in transition or episode data".format(k))
> ```
> - 键 k 不在 `transition_data` 和 `episode_data` 中，抛出异常
>
> ------
>
> 4. 转换数据类型并检查安全视图
> ```python
> dtype = self.scheme[k].get("dtype", th.float32)
> v = th.tensor(v, dtype=dtype, device=self.device)
> self._check_safe_view(v, target[k][_slices])
> ```
> - 获取数据的目标数据类型，默认为 th.float32
> - 将数据 v 转换为指定数据类型的张量，并移动到指定设备
> - 检查数据 v 是否可以安全地视图转换为目标位置的形状
>
>
> 5. 更新目标数据
>
> ```python
> target[k][_slices] = v.view_as(target[k][_slices])
> ```
> - 将数据 v 视图转换为目标位置的形状，并更新目标数据
>
> 6. 预处理数据
>
> ```python
> if k in self.preprocess:
>     new_k = self.preprocess[k][0]
>     v = target[k][_slices]
>     for transform in self.preprocess[k][1]:
>         v = transform.transform(v)
>     target[new_k][_slices] = v.view_as(target[new_k][_slices])
> ```
>
> - 如果键 k 在预处理字典`self.preprocess`中，进行预处理
>   - 获取预处理后的新键`new_k`、目标数据`v`
>   - 遍历`self.preprocess[k][1]`所有预处理进行数据转换
>   - 将预处理后的数据更新到目标位置`target[new_k][_slices]`

> [!TIP]
>
> **总结**
>
> 根据批次切片`bs`和时间步长切片`ts`，将对应的数据分别更新到`ep_batch.data.transition_data`和`ep_batch.data.episode_data`的每个`scheme`中的特征的全空间张量`torch.Size([5000, 61, (k.vshape)])`中的`[(0, 1), (0, 61), (k.vshape)]`和



#### class ReplayBuffer

```python
class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
                super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
```

> [!NOTE]
>
> **调用了父类 EpisodeBatch 的构造函数，并传递了相应的参数**
>
> `class ReplayBuffer` 继承自 `class EpisodeBatch`，初始化 `ReplayBuffer` 对象时，先初始化其父类 `EpisodeBatch` 的属性和方法

> [!TIP]
>
> ```python
> self.buffer_size = buffer_size # same as self.batch_size but more explicit
> ```
>
> 声明子类的self.buffer_index**等同于**父类的self.batch_size

```python
class ReplayBuffer(EpisodeBatch):
    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])
```

> [!NOTE]
>
> **用于将一个新的 `ep_batch` 插入到缓冲区中，由`run.py`中的`run_sequential()`在执行完一批episode后调用**
>
> 1. 检查缓冲区剩余空间
>
>    - 如果当前缓冲区索引 `self.buffer_index` 加上 `ep_batch` 的批次大小 `ep_batch.batch_size` 小于等于缓冲区大小 `self.buffer_size`，则说明缓冲区有足够的空间来插入整个 `ep_batch`。
>
> 2. 插入数据
>
>    - 使用 `self.update` 方法将 `ep_batch` 的 `transition_data` 和 `episode_data` 插入到缓冲区中，插入位置从 `self.buffer_index` 开始
>    - 更新 `self.buffer_index`，使其指向下一个可插入数据的位置
>    - 更新 `self.episodes_in_buffer`，记录缓冲区中存储的批次数据的数量
>    - 确保 `self.buffer_index` 不超过缓冲区大小，通过取模操作 `self.buffer_index % self.buffer_size`
>
> 3. 处理缓冲区空间不足的情况
>
>    - 如果缓冲区剩余空间不足以插入整个 `ep_batch`，则计算剩余空间 `buffer_left`
>
>      递归调用 `insert_episode_batch` 方法，将 `ep_batch` 分成两部分分别插入：先插入前 `buffer_left` 部分，再插入剩余部分





```python
class ReplayBuffer(EpisodeBatch):
    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]
```

> [!NOTE]
>
> - 如果满足 `self.can_sample`条件（ `self.episodes_in_buffer >= batch_size` ），则从缓冲区 `buffer` 随机选择`batch_size`个episodes 
> - 返回 `self[:batch_size]` 和 `self[ep_ids]` ，这些操作会调用 `EpisodeBatch` 类的 `__getitem__` 方法，返回一个新的 `EpisodeBatch` 对象。详见本页开头。


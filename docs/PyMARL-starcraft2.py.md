# PyMARL-starcraft2.py

以qmix | MarineMicro_MvsM_4为例

```python
--config=qmix --env-config=sc2 with env_args.map_name=MarineMicro_MvsM_4 t_max=2050000
```

### starcraft2.py

#### class StarCraft2Env







```python
class StarCraft2Env(MultiAgentEnv):
    def __init__(
        self,
        map_name="8m",
        step_mul=8,
        move_amount=2,
        difficulty="7",
        game_version=None,
        seed=None,
        continuing_episode=False,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=False,
        obs_pathing_grid=False,
        obs_terrain_height=False,
        obs_instead_of_state=False,
        obs_timestep_number=False,
        state_last_action=True,
        state_timestep_number=False,
        reward_sparse=False,
        reward_only_positive=True,
        reward_death_value=10,
        reward_win=200,
        reward_defeat=0,
        reward_negative_scale=0.5,
        reward_scale=True,
        reward_scale_rate=20,
        replay_dir="",
        replay_prefix="",
        window_size_x=1920,
        window_size_y=1200,
        heuristic_ai=False,
        heuristic_rest=False,
        debug=False,
    ):
```

> [!NOTE]
>
> 

> [!TIP]
>
> `agents`：`dict<Unit>` 我方单位字典
>
> battles_game：int 对战局数
>
> battles_won：int 对战胜利局数
>
> death_tracker_ally：ndarray 我方阵亡情况矩阵
>
> death_tracker_enemy：ndarray 敌方阵亡情况矩阵
>
> enemies：`dict<Unit>` 敌方单位字典
>
> heuristic_ai：bool 是否启用启发式动作
>
> last_action：ndarray 上一次动作矩阵
>
> max_reward：float 最高累积奖励

> [!IMPORTANT]
>
> `class Unit`
>
> alliance：int 单位所属阵营
>
> engaged_target_tag：int
>
> health：float 当前生命值
>
> health_max：float 最大生命值
>
> is_active：bool 是否可行动
>
> orders：
>
> pos：Point(x, y, z) 绝对位置
>
> tag：int 单位标签
>
> unit_type：int 单位类别
>
> weapon_cooldown：武器攻击冷却



```python
class StarCraft2Env(MultiAgentEnv):
    def reset(self):
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()
```

> [!NOTE]
>
> **重置当前episode的步数`_episode_steps`**
>
> **根据是否为第一个episode，执行环境启动或重启**
>
> `_episode_steps`：当前episode的步数
>
> `_episode_count`：episode的计数

```python
class StarCraft2Env(MultiAgentEnv):
    def reset(self):		
        ***
    	# Information kept for counting the reward
		self.death_tracker_ally = np.zeros(self.n_agents)
		self.death_tracker_enemy = np.zeros(self.n_enemies)
		self.previous_ally_units = None
		self.previous_enemy_units = None
		self.win_counted = False
		self.defeat_counted = False
```

> [!NOTE]
>
> **为计算奖励保留的信息**





------



```python
class StarCraft2Env(MultiAgentEnv):
    def _restart(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        """
        try:
            self._kill_all_units()
            self._controller.step(2)
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
```

> [!NOTE]
>
> **环境重启方法：通过kill所有单元以重启starcraft环境；如失败，则完全重启环境**

```python
class StarCraft2Env(MultiAgentEnv):
	def init_units(self):
    	"""Initialise the units."""
        
        if all_agents_created and all_enemies_created:  # all good
            return
```

> [!NOTE]
>
> 初始化单位并更新到`self.agents`与`self.enemies`中
>
> 检查`self.agents`与`self.enemies`的长度是否与`self.n_agents`与`self.n_enemies`相同，相同则返回



------

#### 一些方法

```python
@staticmethod
def distance(x1, y1, x2, y2):
    """Distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)
```

> [!NOTE]
>
> **获取两点之间的距离**

```python
class StarCraft2Env(MultiAgentEnv):
	def unit_shoot_range(self, agent_id):
    	"""Returns the shooting range for an agent."""
    	return 6
```

> [!NOTE]
>
> **获取攻击范围，始终为6**

```python
class StarCraft2Env(MultiAgentEnv):
	def unit_max_cooldown(self, unit):
    	"""Returns the maximal cooldown for a unit."""
    	return switcher.get(unit.unit_type, 15)
```

> [!NOTE]
>
> **根据单位种类获取单位冷却的最大值，默认值为15**

```python
class StarCraft2Env(MultiAgentEnv):
	def check_bounds(self, x, y):
    	"""Whether a point is within the map bounds."""
    	return 0 <= x < self.map_x and 0 <= y < self.map_y
```

> [!NOTE]
>
> **坐标点是否在地图边界内**

```python
class StarCraft2Env(MultiAgentEnv):
	def get_state_dict(self):
        state = {"allies": ally_state, "enemies": enemy_state}

        if self.state_last_action:
            state["last_action"] = self.last_action
        if self.state_timestep_number:
            state["timestep"] = self._episode_steps / self.episode_limit
        return state
```

> [!NOTE]
>
> **获取状态**







> [!WARNING]
>
> 1. agent每次执行的动作会从 `avail_actions` 中选取，`avail_actions` 长度 `len(avail_actions) = 1 + 1 + 4 + n_enemies`，代表 `[no_op, stop, move*4, attack*n_enemies]` 。
> 2. 当智能体周围无敌方单位时，`avail_actions` 不会涉及攻击动作
> 3. 当智能体阵亡时，`avail_actions` 仅涉及no_op


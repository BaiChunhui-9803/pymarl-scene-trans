# PyMARL平台修改记录

## 一、软件包文件

- [ ] **注册地图**

  仿照smac_maps.py进行地图注册

  1. ```
     修改 smac.env.starcraft2.maps.__init__.py
     ```

     > [!IMPORTANT]
     >
     > 绝对路径：
     >
     > ```
     > D:\anaconda3\envs\pymarl37\Lib\site-packages\smac\env\starcraft2\maps\__init__.py
     > ```
     >
     > 修改内容：
     >
     > ```python
     > # 自行注册的地图参数
     > from smac.env.starcraft2.maps import user_maps
     > 
     > def get_map_params(map_name):
     >     # map_param_registry = smac_maps.get_smac_map_registry()
     >     map_param_registry = user_maps.get_smac_map_registry()
     >     return map_param_registry[map_name]
     > ```

     

  2. ```
     创建 smac.env.starcraft2.maps.user_maps.py
     ```

     > [!IMPORTANT]
     >
     > 绝对路径：
     >
     > 

  3. 

## 二、用户文件







## 三、注意事项

> [!NOTE]
>
> ```python
> from smac.env.starcraft2.maps import user_maps
> class USERMap(lib.Map):
>     directory = "Example"
>     # download = "https://github.com/oxwhirl/smac#smac-maps"
>     players = 2
>     step_mul = 10
>     game_steps_per_episode = 0 
> ```
>
> 先前在pysc2中，game_steps_per_episode一般给定了固定值，episode达到运行最大步数后会被kill
>
> 而在这里， game_steps_per_episode = 0可以被设置为0，即episode永远不会被kill
>
> **经测试，在pysc2中设置game_steps_per_episode = 0，可以保证episode永远不会被kill，即省去了终端关闭与开启的耗时，在之后的实验中应该利用该特性以加速训练**

> [!NOTE]
>
> ```python
> from smac.env.starcraft2.maps import smac_maps
> map_param_registry = {
>     "3m": {
>         "n_agents": 3,
>         "n_enemies": 3,
>         "limit": 60,
>         "a_race": "T",
>         "b_race": "T",
>         "unit_type_bits": 0,
>         "map_type": "marines",
>     },
>     "MMM": {
>         "n_agents": 10,
>         "n_enemies": 10,
>         "limit": 150,
>         "a_race": "T",
>         "b_race": "T",
>         "unit_type_bits": 3,
>         "map_type": "MMM",
>     },
> }
> ```
>
> ```python
> """
>   n_agents: int 我方智能体个数
>   n_enemies: int 敌方智能体个数
>   limit: int 时间限制？
>   a_race: "T/P/Z" 我方种族
>   b_race: "T/P/Z" 敌方种族
>   unit_type_bits: int 单位类型位数（0即为同构，2/3即为异构）
>   map_type: str 地图类型（实际是地图中单位的类型）
> """
> ```
>
> 





## 四、地图编辑器相关

> [!NOTE]
>
> Event
>
> | 类型  | 标签Label | 名称Name  |      |
> | ----- | --------- | --------- | ---- |
> | Event | Unit      | Unit Dies |      |

> This event fires when a unit dies.  Use "Damage Source Position" to get the position of the unit that dealt the killing blow.  Use "Killing Player" to get the owner of the unit that dealt the killing blow.  Use "Killing Unit" to get the unit that dealt the killing blow.  Use "Triggering Death Type Check" to get the death type.  Use "Triggering Player" to get the owner of the unit that died.  Use "Triggering Unit" to get the unit that died.  Use "Experience Level Of Unit" to get the experience level of the unit that died.  Use "Total Experience Of Unit" to get the total number of experience points the unit had before it died.

似乎是可以获取造成致命伤害的来源，如有需要可以关注
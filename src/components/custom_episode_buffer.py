import torch as th
import numpy as np
from types import SimpleNamespace as SN

class CustomEpisodeCBSBatch:
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

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {},
        })

        for field_key, field_info in scheme.items():
            episode_const = field_info.get("episode_const", False)

            if episode_const:
                self.data.episode_data[field_key] = [[None for _ in range(max_seq_length)] for _ in range(batch_size)]
            else:
                self.data.transition_data[field_key] = [[None for _ in range(max_seq_length)] for _ in range(batch_size)]

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=0, ts=0, mark_filled=True):
        batch_id = bs
        t_id = ts
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][batch_id][t_id] = 1
                    mark_filled = False
            elif k in self.data.episode_data:
                target = self.data.episode_data
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            target[k][batch_id][t_id] = v


    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = CustomEpisodeCBSBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = CustomEpisodeCBSBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())


class CustomReplayBuffer(CustomEpisodeCBSBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, buffer_pool=[], preprocess=None, device="cpu"):
        super(CustomReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.buffer_pool = buffer_pool

    def retain_filled(self, ep_batch):
        filled = ep_batch.data.transition_data["filled"]
        for i in range(ep_batch.batch_size):
            filled_i = filled[i]
            max_t = filled_i.index(None) if None in filled_i else len(filled_i)
            for k, v in ep_batch.data.transition_data.items():
                # if k == "filled":
                #     continue
                v[i] = v[i][:max_t]
            for k, v in ep_batch.data.episode_data.items():
                v[i] = v[i][:max_t]

    def insert_episode_batch(self, ep_batch):
        # 除去ep_batch中，data.transition_data["filled"]为None的数据
        self.retain_filled(ep_batch)
        self.buffer_index = self.buffer_index % self.buffer_size
        if len(self.buffer_pool) < self.buffer_size:
            self.buffer_pool.append(ep_batch)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index += ep_batch.batch_size
        else:
            self.buffer_pool[self.buffer_index] = ep_batch
            self.buffer_index += ep_batch.batch_size

        # if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
        #     # 将ep_batch中的数据插入到buffer_pool中
        #     self.buffer_pool.append(ep_batch)
        #     self.buffer_index += ep_batch.batch_size
        #     self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
        #     self.buffer_index = self.buffer_index % self.buffer_size
        #     assert self.buffer_index < self.buffer_size



    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
        sample_buffer = [self.buffer_pool[id] for id in ids]
        if "filled" in self.scheme:
            self.scheme.pop("filled")
        return CustomReplayBuffer(self.scheme, self.groups, batch_size, self.max_seq_length,
                                  buffer_pool=sample_buffer, device=self.device)


    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())


import json
import random


def train_to_test(trains, tests, process_flag):
    if process_flag == "avg_win_rate":

        return sum(trains) / len(trains), sum(tests) / len(tests)
    elif process_flag == "avg_final_score":
        return sum(trains) / len(trains), sum(tests) / len(tests)


class DataReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config_data = None
        self.info_data = None
        self.HRL_data = None

    def read(self):
        # 检查文件路径是否存在
        try:
            with open(self.file_path + 'config.json', 'r') as f:
                self.config_data = json.loads(f.read())
        except FileNotFoundError:
            print("\033[91m{}: File {} not found\033[0m".format(self.file_path, 'config.json'))
            return False
        try:
            with open(self.file_path + 'info.json', 'r') as f:
                self.info_data = json.loads(f.read())
        except FileNotFoundError:
            print("\033[91m{}: File {} not found\033[0m".format(self.file_path, 'info.json'))
            return False
        return True

    def read_HRL(self):
        with open(self.file_path + 'game_result.txt', 'r') as f:
            win_state = []
            final_score = []
            for line in f.readlines():
                win_state.append(line.strip().split('\t')[0])
                final_score.append(int(line.split()[2]) + int(line.split()[3]) if line.split()[0] == 'Win'
                                   else int(line.split()[2]) * random.randint(8, 10) / 10 + int(line.split()[3]))
            self.HRL_data = {'win_state': win_state, 'final_score': final_score}
            win_rate = []
            for i in range(len(win_state)):
                start_index = max(0, i - 39)  # 计算起始节点的索引
                win_count = win_state[start_index:i + 1].count("Win")  # 统计第i-20到第i个节点中"Win"的次数
                win_rate.append(win_count / (i - start_index + 2))
            self.HRL_data['win_rate'] = win_rate
            avg_score = []
            for i in range(len(final_score)):
                start_index = max(0, i - 39)
                avg_score.append(sum(final_score[start_index:i + 1]) / (i - start_index + 2))
            self.HRL_data['avg_score'] = avg_score
            # print(self.HRL_data['win_rate'])
            # print(self.HRL_data['avg_score'])

    def config_data_filter(self, fields):
        config_filter_dict = {}
        for field in fields:
            if field in self.config_data:
                config_filter_dict[field] = self.config_data[field]
            elif field.split('.')[0] in self.config_data:
                config_filter_dict[field] = self.config_data[field.split('.')[0]][field.split('.')[1]]
            else:
                pass
        return config_filter_dict

    def info_data_filter(self, fields):
        info_filter_dict = {}
        for field in fields:
            if field in self.info_data:
                info_filter_dict[field] = self.info_data[field]
            else:
                pass
        return info_filter_dict
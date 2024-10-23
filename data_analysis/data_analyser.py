from data_reader import DataReader, train_to_test
import matplotlib.pyplot as plt
import numpy as np

class DataAnalyser:
    def __init__(self, data):
        self.data = data

    def analyse(self):
        print("Analysing data: ", self.data)

data_HRL_path = {
    "MarineMicro_MvsM_4": {
        "trains": {
            "path0": "../results_HRL/MM4/trains/0/",
            "path1": "../results_HRL/MM4/trains/1/",
            "path2": "../results_HRL/MM4/trains/3/",
            "path3": "../results_HRL/MM4/trains/5/",
            "path4": "../results_HRL/MM4/trains/9/",
        },
        "tests": {
            "path0": "../results_HRL/MM4/tests/0/",
            "path1": "../results_HRL/MM4/tests/1/",
            "path2": "../results_HRL/MM4/tests/3/",
            "path3": "../results_HRL/MM4/tests/5/",
            "path4": "../results_HRL/MM4/tests/9/",
        }
    },
    "MarineMicro_MvsM_4_mirror": {
        "trains": {
            "path0": "../results_HRL/MM4m/trains/0/",
            "path1": "../results_HRL/MM4m/trains/1/",
            "path2": "../results_HRL/MM4m/trains/3/",
            "path3": "../results_HRL/MM4m/trains/6/",
        },
        "tests": {
            "path0": "../results_HRL/MM4m/tests/0/",
            "path1": "../results_HRL/MM4m/tests/1/",
            "path2": "../results_HRL/MM4m/tests/3/",
            "path3": "../results_HRL/MM4m/tests/6/",
        }
    },
    "MarineMicro_MvsM_4_dist": {
        "trains": {
            "path0": "../results_HRL/MM4dist/trains/2/",
            "path1": "../results_HRL/MM4dist/trains/4/",
            "path2": "../results_HRL/MM4dist/trains/5/",
            "path3": "../results_HRL/MM4dist/trains/7/",
            "path4": "../results_HRL/MM4dist/trains/8/",
        },
        "tests": {
            "path0": "../results_HRL/MM4dist/tests/2/",
            "path1": "../results_HRL/MM4dist/tests/4/",
            "path2": "../results_HRL/MM4dist/tests/5/",
            "path3": "../results_HRL/MM4dist/tests/7/",
            "path4": "../results_HRL/MM4dist/tests/8/",
        }
    },
    "MarineMicro_MvsM_4_dist_mirror": {
        "trains": {
            "path0": "../results_HRL/MM4distm/trains/0/",
            "path1": "../results_HRL/MM4distm/trains/1/",
            "path2": "../results_HRL/MM4distm/trains/2/",
            "path3": "../results_HRL/MM4distm/trains/3/",
        },
        "tests": {
            "path0": "../results_HRL/MM4distm/tests/0/",
            "path1": "../results_HRL/MM4distm/tests/1/",
            "path2": "../results_HRL/MM4distm/tests/2/",
            "path3": "../results_HRL/MM4distm/tests/3/",
        }
    },
    "MarineMicro_MvsM_8": {
        "trains": {
            "path0": "../results_HRL/MM8/trains/0-hn/",
            "path1": "../results_HRL/MM8/trains/4/",
            "path2": "../results_HRL/MM8/trains/5/",
        },
        "tests": {
            "path0": "../results_HRL/MM8/tests/0-hn/",
            "path1": "../results_HRL/MM8/tests/4/",
            "path2": "../results_HRL/MM8/tests/5/",
        }
    },
    "MarineMicro_MvsM_8_mirror": {
        "trains": {
            "path0": "../results_HRL/MM8m/trains/0-hn/",
            "path1": "../results_HRL/MM8m/trains/2-hn/",
            "path2": "../results_HRL/MM8m/trains/3-hn/",
        },
        "tests": {
            "path0": "../results_HRL/MM8m/tests/0-hn/",
            "path1": "../results_HRL/MM8m/tests/2-hn/",
            "path2": "../results_HRL/MM8m/tests/3-hn/",
        }
    },
}

data_path = {
    "MarineMicro_MvsM_4": {
        "qmix": "../results_saved/qmix t_max=2050000/MarineMicro_MvsM_4/",
        "coma": "../results_saved/coma t_max=2050000/MarineMicro_MvsM_4/",
        "iql": "../results_saved/iql t_max=2050000/MarineMicro_MvsM_4/",
        "vdn": "../results_saved/vdn t_max=2050000/MarineMicro_MvsM_4/",
        "qtran": "../results_saved/qtran t_max=2050000/MarineMicro_MvsM_4/",
    },
    "MarineMicro_MvsM_4_mirror": {
        "qmix": "../results_saved/qmix t_max=2050000/MarineMicro_MvsM_4_mirror/",
        "coma": "../results_saved/coma t_max=2050000/MarineMicro_MvsM_4_mirror/",
        "iql": "../results_saved/iql t_max=2050000/MarineMicro_MvsM_4_mirror/",
        "vdn": "../results_saved/vdn t_max=2050000/MarineMicro_MvsM_4_mirror/",
        "qtran": "../results_saved/qtran t_max=2050000/MarineMicro_MvsM_4_mirror/",
    },
    "MarineMicro_MvsM_4_dist": {
        "qmix": "../results_saved/qmix t_max=2050000/MarineMicro_MvsM_4_dist/",
        "coma": "../results_saved/coma t_max=2050000/MarineMicro_MvsM_4_dist/",
        "iql": "../results_saved/iql t_max=2050000/MarineMicro_MvsM_4_dist/",
        "vdn": "../results_saved/vdn t_max=2050000/MarineMicro_MvsM_4_dist/",
        "qtran": "../results_saved/qtran t_max=2050000/MarineMicro_MvsM_4_dist/",
    },
    "MarineMicro_MvsM_4_dist_mirror": {
        "qmix": "../results_saved/qmix t_max=2050000/MarineMicro_MvsM_4_dist_mirror/",
        "coma": "../results_saved/coma t_max=2050000/MarineMicro_MvsM_4_dist_mirror/",
        "iql": "../results_saved/iql t_max=2050000/MarineMicro_MvsM_4_dist_mirror/",
        "vdn": "../results_saved/vdn t_max=2050000/MarineMicro_MvsM_4_dist_mirror/",
        "qtran": "../results_saved/qtran t_max=2050000/MarineMicro_MvsM_4_dist_mirror/",
    },
    "MarineMicro_MvsM_8": {
        "qmix": "../results_saved/qmix t_max=2050000/MarineMicro_MvsM_8/",
        "coma": "../results_saved/coma t_max=2050000/MarineMicro_MvsM_8/",
        "iql": "../results_saved/iql t_max=2050000/MarineMicro_MvsM_8/",
        "vdn": "../results_saved/vdn t_max=2050000/MarineMicro_MvsM_8/",
        "qtran": "../results_saved/qtran t_max=2050000/MarineMicro_MvsM_8/",
    },
    "MarineMicro_MvsM_8_mirror": {
        "qmix": "../results_saved/qmix t_max=2050000/MarineMicro_MvsM_8_mirror/",
        "coma": "../results_saved/coma t_max=2050000/MarineMicro_MvsM_8_mirror/",
        "iql": "../results_saved/iql t_max=2050000/MarineMicro_MvsM_8_mirror/",
        "vdn": "../results_saved/vdn t_max=2050000/MarineMicro_MvsM_8_mirror/",
        "qtran": "../results_saved/qtran t_max=2050000/MarineMicro_MvsM_8_mirror/",
    },
}

data_fields = {
    "config_fields": [
        "env_args.map_name",
        "t_max",
        "name",
    ],
    "info_fields": [
        "episode",
        "battle_won_mean",
        "dead_allies_mean",
        "dead_enemies_mean",
        "sum_health_agents_mean",
        "sum_health_enemies_mean",
        "ep_length_mean",
        "test_battle_won_mean",
        "test_dead_allies_mean",
        "test_dead_enemies_mean",
        "test_sum_health_agents_mean",
        "test_sum_health_enemies_mean",
        "test_ep_length_mean",
    ],
}

train_metrics = [
    "episode",
    "train_win_rate",
    "train_final_score",
]

test_metrics = [
    "episode",
    "test_win_rate",
    "test_final_score",
]

if __name__ == "__main__":
    # 读取results_saved文件夹下的config.json和info.json文件
    # data_reader = DataReader(data_path.get("MarineMicro_MvsM_4").get("qmix"))
    # config_data, info_data = data_reader.read()
    # print(data_reader.config_data_filter(data_fields["config_fields"]))
    # print(data_reader.info_data_filter(data_fields["info_fields"]))
    data = {}
    for scene in data_path.keys():
        data[scene] = {}
        for alg in data_path[scene].keys():
            data[scene][alg] = {}
            data_reader = DataReader(data_path.get(scene).get(alg))
            # print("scene: ", data_reader)
            if data_reader.read():
                data[scene][alg][train_metrics[0]] = data_reader.info_data_filter(data_fields["info_fields"])["episode"]
                data[scene][alg][train_metrics[1]] = data_reader.info_data_filter(data_fields["info_fields"])["battle_won_mean"]
                data[scene][alg][train_metrics[2]] = [sa - se for sa, se in zip(data_reader.info_data_filter(data_fields["info_fields"])["sum_health_agents_mean"],
                                                                                data_reader.info_data_filter(data_fields["info_fields"])["sum_health_enemies_mean"])]
                data[scene][alg][test_metrics[0]] = data_reader.info_data_filter(data_fields["info_fields"])["episode"]
                data[scene][alg][test_metrics[1]] = data_reader.info_data_filter(data_fields["info_fields"])["test_battle_won_mean"]
                data[scene][alg][test_metrics[2]] = [sa - se for sa, se in zip(data_reader.info_data_filter(data_fields["info_fields"])["test_sum_health_agents_mean"],
                                                                                data_reader.info_data_filter(data_fields["info_fields"])["test_sum_health_enemies_mean"])]


    data_HRL = {}
    for scene in data_HRL_path.keys():
        data_HRL[scene] = {}
        for process in data_HRL_path[scene].keys():
            data_HRL[scene][process] = {}
            avg_win_rate_dict = {}
            avg_final_score_dict = {}
            for path in data_HRL_path[scene][process].values():
                data_reader = DataReader(path)
                data_reader.read_HRL()
                avg_win_rate_dict[path.split('/')[-2]] = data_reader.HRL_data["win_rate"]
                avg_final_score_dict[path.split('/')[-2]] = data_reader.HRL_data["avg_score"]
                # print(data_reader.HRL_data["avg_score"])
            data_HRL[scene][process]["avg_win_rate"] = [sum(values) / len(values) for values in zip(*avg_win_rate_dict.values())]
            data_HRL[scene][process]["avg_final_score"] = [sum(values) / len(values) for values in zip(*avg_final_score_dict.values())]
    # print(data_HRL.get("MarineMicro_MvsM_4").get("trains").get("avg_win_rate"))

    # print(data)

    for scene in data_HRL.keys():
        for process in data_HRL[scene].keys():
            for metric in data_HRL[scene][process].keys():
                print(f"{scene} {process} {metric}: {data_HRL[scene][process][metric]}")

            # avg_win_rate = [sum(win_rate) / len(win_rate) for win_rate in avg_win_rate_dict.values()]
            # print(avg_win_rate)
                # print(data_reader.HRL_data["win_rate"])



    for scene in data.keys():
        fig, axs = plt.subplots(2, 2, figsize=(16, 8))
        # fig.suptitle(f"Metrics for {scene}", fontsize=16)

        for i, metric in enumerate(train_metrics[1:]):
            max_episode = 500
            data_HRL_len = 500
            for alg in data[scene].keys():
                if data[scene][alg]:
                    episodes = data[scene][alg]["episode"]
                    metric_data = data[scene][alg][metric]
                    min_length = min(len(episodes), len(metric_data))
                    if episodes[-1] > max_episode:
                        max_episode = episodes[-1]
                    axs[i, 0].plot(episodes[:min_length], metric_data[:min_length], label=f"{alg}")
            axs[i, 0].set_xlim(0, max_episode)
            # 制第二个红色的x轴在原x轴下方，取值是0-500
            # if scene in data_HRL and "trains" in data_HRL[scene]:
            #     axs2 = axs[i, 0].twiny()
            #     if len(data_HRL[scene]["trains"][f"avg_win_rate"]) <= 500:
            #         data_HRL_len = 500
            #     elif len(data_HRL[scene]["trains"][f"avg_win_rate"]) >= 1499:
            #         data_HRL_len = 1500
            #     axs2.set_xlim(0, data_HRL_len)
            #     axs2.spines['top'].set_color('red')
            #     axs2.tick_params(axis='x', direction='in', width=1, colors='red')
            #     if metric == "train_win_rate":
            #         axs2.plot(data_HRL[scene]["trains"][f"avg_win_rate"], label="HRL avg", color="r")
            #     elif metric == "train_final_score":
            #         axs2.plot(data_HRL[scene]["trains"][f"avg_final_score"], label="HRL avg", color="r")
            axs[i, 0].legend(loc="lower right")
            axs[i, 0].set_title(f"{metric}")
            axs[i, 0].set_xlabel("Episodes")
            axs[i, 0].set_ylabel(metric)

        for i, metric in enumerate(test_metrics[1:]):
            max_episode = 500
            data_HRL_len = 500
            for alg in data[scene].keys():
                if data[scene][alg]:
                    episodes = data[scene][alg]["episode"]
                    metric_data = data[scene][alg][metric]
                    min_length = min(len(episodes), len(metric_data))
                    if episodes[-1] > max_episode:
                        max_episode = episodes[-1]
                    axs[i, 1].plot(episodes[:min_length], metric_data[:min_length], label=f"{alg}")
            axs[i, 1].set_xlim(0, max_episode)
            # if scene in data_HRL and "tests" in data_HRL[scene]:
            #     axs2 = axs[i, 1].twiny()
            #     if len(data_HRL[scene]["tests"][f"avg_win_rate"]) <= 500:
            #         data_HRL_len = 500
            #     elif len(data_HRL[scene]["tests"][f"avg_win_rate"]) >= 1499:
            #         data_HRL_len = 1500
            #     axs2.set_xlim(0, data_HRL_len)
            #     axs2.spines['top'].set_color('red')
            #     axs2.tick_params(axis='x', direction='in', width=1, colors='red')
            #     if metric == "test_win_rate":
            #         axs2.plot(data_HRL[scene]["tests"][f"avg_win_rate"], label="HRL avg", color="r")
            #     elif metric == "test_final_score":
            #         axs2.plot(data_HRL[scene]["tests"][f"avg_final_score"], label="HRL avg", color="r")
                # if metric == "test_win_rate":
                #     axs2.plot(train_to_test(data_HRL[scene]["trains"], data_HRL[scene]["tests"], "avg_win_rate"), label="HRL avg", color="r")
                # elif metric == "test_final_score":
                #     axs2.plot(train_to_test(data_HRL[scene]["trains"], data_HRL[scene]["tests"], "avg_final_score"), label="HRL avg", color="r")
            axs[i, 1].legend(loc="lower right")
            axs[i, 1].set_title(f"{metric}")
            axs[i, 1].set_xlabel("Episodes")
            axs[i, 1].set_ylabel(metric)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.show()
        plt.savefig(f"data_figs/{scene}.pdf")
    # range(len(data_HRL[scene]["tests"][f"avg_win_rate"]

    # data1_list = data_reader.info_data_filter(data_fields["info_fields"])["battle_won_mean"]
    # print(data1_list)
    # print("config_data: ", config_data)
    # print("info_data: ", info_data)
    # data_analyser = DataAnalyser(config_data)
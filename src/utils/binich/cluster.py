
from sklearn.cluster import KMeans
import numpy as np
import math

cluster_strengths = {
    "k_means_000": 0.0,
    "k_means_025": 0.25,
    "k_means_050": 0.5,
    "k_means_075": 0.75,
    "k_means_100": 1.0
}

def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

class Cluster:
    def __init__(self, n_agents):
        self.cluster_data = None
        self.agents = None
        self.enemies = None
        self.sorted_agents = None
        self.sorted_enemies = None
        self.featured_agents = None
        self.featured_enemies = None
        self.alive_agents = None
        self.alive_enemies = None
        self.name = None
        self.n_agents = n_agents
        self.cluster_strengths = cluster_strengths

        self._unit_shoot_range = 5

    def get_shoot_range(self):
        return self._unit_shoot_range

    def update(self, agents, enemies):
        self.agents = agents
        self.enemies = enemies
        self.sorted_agents = [{'tag': agent.tag, 'x': agent.pos.x, 'y': agent.pos.y, 'health': agent.health}
                              for agent in self.agents.values()]
        self.sorted_enemies = [{'tag': enemy.tag, 'x': enemy.pos.x, 'y': enemy.pos.y, 'health': enemy.health}
                               for enemy in self.enemies.values()]
        self.featured_agents = sorted(
            [(item['tag'], item['x'], item['y'], item['health']) for item in self.sorted_agents], key=lambda x: x[0])
        self.featured_enemies = sorted(
            [(item['tag'], item['x'], item['y'], item['health']) for item in self.sorted_enemies], key=lambda x: x[0])
        self.alive_agents = [agent for agent in self.featured_agents if agent[3] > 0]
        self.alive_enemies = [enemy for enemy in self.featured_enemies if enemy[3] > 0]

    def kmeans(self, k):
        x = []
        for point in self.alive_agents:
            x.append(point[1:])
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x)
        labels = kmeans.predict(x)
        clustered_points = []
        for i in range(len(self.alive_agents)):
            clustered_points.append((self.alive_agents[i] + (labels[i],)))
        self.cluster_data = clustered_points
        return self.cluster_data

    def k_means_000(self):
        units = self.alive_agents
        clu_number = len(units)
        clu_lists = [clu_number, 0., []]
        if clu_number > 0:
            clu_variance = self.calculate_variance_sum(units)
            clu_lists[1] = round(clu_variance * 2) / 2
            for i in range(clu_number):
                clu_lists[2].append((
                    i, (units[i][1], units[i][2]), 0.0, 1.0, [units[i]]
                ))
        return clu_lists

    def k_means_025(self):
        units = self.alive_agents
        clu_number = 1
        clu_lists = [clu_number, 0., []]
        if len(units) * 0.25 > 1:
            clu_number = int(len(units) * 0.25)
            clu_lists[0] = clu_number
        if len(units) > 0:
            clusters = self.kmeans(clu_number)
            clu_center_list = []
            unique_labels = set([cluster[-1] for cluster in clusters])
            for label in unique_labels:
                cluster_points = [cluster[:-1] for cluster in clusters if cluster[-1] == label]  # 获取具有相同聚类标签的坐标点
                clu_uniformity = self.calculate_clu_uniformity(cluster_points)
                clu_crowding = self.calculate_clu_crowding(cluster_points, self._unit_shoot_range)
                clu_0_center, clu_0_radius = self.circle_fitting(cluster_points, 0.)
                clu_center_list.append((0, clu_0_center[0], clu_0_center[1]))
                clu_lists[2].append((
                    label, clu_0_center, clu_uniformity, clu_crowding, cluster_points
                ))
            clu_variance = self.calculate_variance_sum(clu_center_list)
            clu_lists[1] = round(clu_variance * 2) / 2
        return clu_lists

    def k_means_050(self):
        units = self.alive_agents
        clu_number = 1
        clu_lists = [clu_number, 0., []]
        if len(units) * 0.5 > 1:
            clu_number = int(len(units) * 0.5)
            clu_lists[0] = clu_number
        if len(units) > 0:
            clusters = self.kmeans(clu_number)
            clu_center_list = []
            unique_labels = set([cluster[-1] for cluster in clusters])
            for label in unique_labels:
                cluster_points = [cluster[:-1] for cluster in clusters if cluster[-1] == label]  # 获取具有相同聚类标签的坐标点
                clu_uniformity = self.calculate_clu_uniformity(cluster_points)
                clu_crowding = self.calculate_clu_crowding(cluster_points, self._unit_shoot_range)
                clu_0_center, clu_0_radius = self.circle_fitting(cluster_points, 0.)
                clu_center_list.append((0, clu_0_center[0], clu_0_center[1]))
                clu_lists[2].append((
                    label, clu_0_center, clu_uniformity, clu_crowding, cluster_points
                ))
            clu_variance = self.calculate_variance_sum(clu_center_list)
            clu_lists[1] = round(clu_variance * 2) / 2
        return clu_lists

    def k_means_075(self):
        units = self.alive_agents
        clu_number = 1
        clu_lists = [clu_number, 0., []]
        if len(units) * 0.5 > 1:
            clu_number = int(len(units) * 0.5)
            clu_lists[0] = clu_number
        if len(units) > 0:
            clusters = self.kmeans(clu_number)
            clu_center_list = []
            unique_labels = set([cluster[-1] for cluster in clusters])
            for label in unique_labels:
                cluster_points = [cluster[:-1] for cluster in clusters if cluster[-1] == label]  # 获取具有相同聚类标签的坐标点
                clu_uniformity = self.calculate_clu_uniformity(cluster_points)
                clu_crowding = self.calculate_clu_crowding(cluster_points, self._unit_shoot_range)
                clu_0_center, clu_0_radius = self.circle_fitting(cluster_points, 0.)
                clu_center_list.append((0, clu_0_center[0], clu_0_center[1]))
                clu_lists[2].append((
                    label, clu_0_center, clu_uniformity, clu_crowding, cluster_points
                ))
            clu_variance = self.calculate_variance_sum(clu_center_list)
            clu_lists[1] = round(clu_variance * 2) / 2
        return clu_lists

    def k_means_100(self):
        units = self.alive_agents
        clu_number = 1
        clu_variance = 0.
        clu_lists = [clu_number, clu_variance, []]
        if len(units) > 0:
            clu_uniformity = self.calculate_clu_uniformity(units)
            clu_crowding = self.calculate_clu_crowding(units, self._unit_shoot_range)
            clu_0_center, clu_0_radius = self.circle_fitting(units, 0.)
            clu_lists[2].append((
                0, clu_0_center, clu_uniformity, clu_crowding, units
            ))
        return clu_lists

    def circle_fitting(self, my_units_lst: list, ex_radius):
        center_x_sum = 0
        center_y_sum = 0
        num_points = len(my_units_lst)
        for point in my_units_lst:
            center_x_sum += point[1]
            center_y_sum += point[2]
        center_point = (center_x_sum / num_points, center_y_sum / num_points)

        dist_max = 0.
        for point in my_units_lst:
            dist = distance((point[1], point[2]), center_point)
            if dist >= dist_max:
                dist_max = dist
        radius = dist_max + ex_radius
        return center_point, radius

    # 计算标准差
    def calculate_std_deviation(self, numbers):
        n = len(numbers)
        mean = sum(numbers) / n
        squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
        variance = squared_diff_sum / n
        std_deviation = math.sqrt(variance)
        return std_deviation

    # 计算变异系数
    def calculate_coefficient_of_variation(self, numbers):
        if len(numbers) > 1:
            mean = sum(numbers) / len(numbers)
            if mean == 0.0:
                return 0.0
            std_deviation = self.calculate_std_deviation(numbers)  # 使用前面提到的计算标准差的函数
            coefficient_of_variation = (std_deviation / mean)
            return coefficient_of_variation
        else:
            return 0.0

    # 圆中散点拥挤度
    def calculate_clu_crowding(self, my_units_lst: list, min_radius):
        center_point, radius = self.circle_fitting(my_units_lst, 0.)
        if radius == 0.0:
            return 1.0
        else:
            total_points = len(my_units_lst)
            total_distance = 0
            crowding = 0.
            if len(my_units_lst):
                if len(my_units_lst) == 1:
                    return 1.0
                else:
                    for i in range(total_points):
                        min_distance = float('inf')
                        for j in range(total_points):
                            if i != j:
                                d = distance((my_units_lst[i][1], my_units_lst[i][2]),
                                             (my_units_lst[j][1], my_units_lst[j][2]))
                                if d < min_distance:
                                    min_distance = d
                        total_distance += min_distance
                    max_distance = 2 * total_points * radius * math.sin(math.pi / total_points)
                    # print(my_units_lst)
                    crowding = 1. - total_distance / max_distance
            else:
                crowding = 0.
            return round(crowding, 2)

    def calculate_clu_uniformity(self, my_units_lst: list):
        center_point, radius = self.circle_fitting(my_units_lst, 0.)
        distances = []
        uniformity = 0.
        if len(my_units_lst):
            if len(my_units_lst) == 1:
                return uniformity
            else:
                for point in my_units_lst:
                    x = point[1]
                    y = point[2]
                    distance = math.sqrt((x - center_point[0]) ** 2 + (y - center_point[1]) ** 2)
                    distances.append(distance)
                uniformity = 1. - self.calculate_coefficient_of_variation(distances)
        else:
            uniformity = 0.
        return round(uniformity, 2)

    def calculate_variance_sum(self, my_units_lst: list):
        x = [point[1] for point in my_units_lst]  # 提取 x 坐标
        y = [point[2] for point in my_units_lst]  # 提取 y 坐标
        x_var = np.var(x)  # 计算 x 坐标的方差
        y_var = np.var(y)  # 计算 y 坐标的方差
        variance_sum = x_var + y_var  # 计算 x、y 方差之和
        return variance_sum

    def hashing(self, cluster_list):
        mapped = ""
        if len(cluster_list) > 0:
            for item in cluster_list:
                mapped += '{:01X}'.format(int(item[0] * 15.9))
                mapped += '{:01X}'.format(int(item[1] * 15.9))
            return mapped
        else:
            return "X"

    def __str__(self):
        return f'Cluster(name={self.name})'

    def __repr__(self):
        return str(self)

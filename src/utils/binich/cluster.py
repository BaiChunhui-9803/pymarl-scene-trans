
from sklearn.cluster import KMeans
import numpy as np

cluster_strengths = {
    "k_means_000": 0.0,
    "k_means_025": 0.25,
    "k_means_050": 0.5,
    "k_means_075": 0.75,
    "k_means_100": 1.0
}


class Cluster:
    def __init__(self, n_agents):
        self.cluster_data = None
        self.agents = None
        self.enemies = None
        self.sorted_agents = None
        self.sorted_enemies = None
        self.featured_agents = None
        self.featured_enemies = None
        self.name = None
        self.n_agents = n_agents
        self.cluster_strengths = cluster_strengths

    def update(self, agents, enemies):
        self.agents = agents
        self.enemies = enemies
        self.sorted_agents = [{'tag': agent.tag, 'x': agent.pos.x, 'y': agent.pos.y} for agent in self.agents.values()]
        self.sorted_enemies = [{'tag': enemy.tag, 'x': enemy.pos.x, 'y': enemy.pos.y} for enemy in self.enemies.values()]
        self.featured_agents = sorted([(item['tag'], item['x'], item['y']) for item in self.sorted_enemies], key=lambda x: x[0])
        self.featured_enemies = sorted([(item['tag'], item['x'], item['y']) for item in self.sorted_agents], key=lambda x: x[0])

    def kmeans(self, k):
        x = []
        for point in self.featured_agents:
            x.append(point[1:])
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x)
        labels = kmeans.predict(x)
        clustered_points = []
        for i in range(len(self.featured_agents)):
            clustered_points.append((self.featured_agents[i] + (labels[i],)))
        self.cluster_data = clustered_points
        return self.cluster_data


    def __str__(self):
        return f'Cluster(name={self.name})'

    def __repr__(self):
        return str(self)

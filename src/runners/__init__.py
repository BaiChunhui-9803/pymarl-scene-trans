REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

# binich
from .custom_episode_runner import EpisodeRunner
REGISTRY["episode_cbs"] = EpisodeRunner
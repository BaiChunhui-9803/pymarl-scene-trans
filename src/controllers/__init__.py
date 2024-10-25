REGISTRY = {}

from .basic_controller import BasicMAC

REGISTRY["basic_mac"] = BasicMAC

# binich
from .custom_controller import CustomController
REGISTRY["custom_controller"] = CustomController
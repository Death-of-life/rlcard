name = "rlcard-ptcg"
__version__ = "2.0.0"

from rlcard.envs import make

try:
    from rlcard.agents import RandomAgent
except ImportError:
    pass


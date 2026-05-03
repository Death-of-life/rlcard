''' Register environments
'''
from rlcard.envs.env import Env
from rlcard.envs.registration import register, make

register(
    env_id='ptcg',
    entry_point='rlcard.envs.ptcg:PtcgEnv',
)

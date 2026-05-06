import subprocess
import sys

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

if 'torch' in installed_packages:
    from rlcard.agents.dqn_agent import DQNAgent as DQNAgent
    from rlcard.agents.nfsp_agent import NFSPAgent as NFSPAgent
    from rlcard.agents.ptcg_ppo_agent import PtcgPPOAgent as PtcgPPOAgent

from rlcard.agents.random_agent import RandomAgent

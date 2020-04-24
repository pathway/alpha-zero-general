import Arena
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game
from connect4.Connect4Players import *
from connect4.tensorflow.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = False # True

g = Connect4Game(6,7,4)

# all players
rp = RandomPlayer(g).play
op = OneStepLookaheadConnect4Player(g).play
hp = HumanConnect4Player(g).play


if 1:
    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./temp/','best.pth.tar')
    args1 = dotdict({'numMCTSSims': 1, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    p1_name = 'az-1'
else:
    n1p = op
    p1_name = 'op'

if human_vs_cpu:
    player2 = hp
    p2_name = 'Hooman'
else:
    n2 = NNet(g)
    n2.load_checkpoint('./temp/','best.pth.tar')
    args2 = dotdict({'numMCTSSims': 500, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    p2_name = 'az-500'

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(p1_name, p2_name, n1p, player2, g, display=Connect4Game.display)

print(arena.playGames(2, verbose=True))

import Arena
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game
from connect4.Connect4Players import *
from connect4.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

human_vs_cpu = True #False # True

g = Connect4Game(6,7,4)

n1 = NNet(g)
n1.load_checkpoint('./temp/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':2.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

def my_agent(observation, configuration):

    #print(observation, configuration)
    # kaggle act function
    mark = observation.mark  # 1 or 2
    curPlayer = mark
    if mark == 2: curPlayer = -1

    columns = configuration.columns

    obs_board_list = observation.board
    obs_board = np.array(obs_board_list)
    obs_board = obs_board.reshape((6, 7))
    obs_board = np.where(obs_board == 2.0, -1, obs_board, )
    #print(obs_board)

    can_board = g.getCanonicalForm(obs_board, curPlayer)

    valids = np.where( g.getValidMoves(g.getCanonicalForm(obs_board, curPlayer), 1) , 1.0, 0.0)
    action_probs = mcts1.getActionProb(can_board, temp=0)
    print(action_probs)
    action_probs *= valids
    action = np.argmax(action_probs)

    print("valids",valids)
    print(type(action))
    return int(action)


def try_agent():
    import numpy as np
    from kaggle_environments import evaluate, make, utils

    env = make("connectx", debug=True)
    env.reset()

    # Play as first position against random agent.
    trainer = env.train([ None, "negamax"])

    observation = trainer.reset()

    while not env.done:
        my_action = my_agent(observation, env.configuration)
        print("My Action", my_action)
        observation, reward, done, info = trainer.step(my_action)
        print(reward)
        # env.render(mode="ipython", width=100, height=90, header=False, controls=False)
    env.render()

    #print(res)
    exit(1)


#try_agent()
def eval_agent():

    agents = ["random", my_agent]
    #agents = ["random", "negamax"] # my_agent]

    # How many times to run them.
    num_episodes = 3
    steps = []
    debug = True
    from kaggle_environments import evaluate, make, utils

    env = make("connectx", debug=False)
    env.reset()

    res = env.run(agents)
    print(res[-1])
    env.render()
    #rewards = evaluate("connectx", agents, env.configuration, steps, num_episodes,)
    #print(rewards)

#try_agent()

eval_agent()

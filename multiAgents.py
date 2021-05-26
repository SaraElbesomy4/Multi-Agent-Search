# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
	"""
	A reflex agent chooses an action at each choice point by examining
	its alternatives via a state evaluation function.

	The code below is provided as a guide.  You are welcome to change
	it in any way you see fit, so long as you don't touch our method
	headers.
	"""


	def getAction(self, gameState):
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)

		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


		"*** YOUR CODE HERE ***"

		newFood = newFood.asList()
		distance_to_food = float("inf")
		for food in newFood:
			d_food = manhattanDistance(newPos, food)
			distance_to_food = min(distance_to_food,d_food)


		ghost_pos = successorGameState.getGhostPositions()
		for ghost in ghost_pos :
			d_ghost = manhattanDistance(newPos, ghost)
			if d_ghost < 5:
				return -float('inf')
		
		return successorGameState.getScore() + 1.0 / distance_to_food + 1.0/d_ghost

def scoreEvaluationFunction(currentGameState):
	"""
	This default evaluation function just returns the score of the state.
	The score is the same one displayed in the Pacman GUI.

	This evaluation function is meant for use with adversarial search agents
	(not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	This class provides some common elements to all of your
	multi-agent searchers.  Any methods defined here will be available
	to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	You *do not* need to make any changes here, but you can if you want to
	add functionality to all your adversarial search agents.  Please do not
	remove anything, however.

	Note: this is an abstract class: one that should not be instantiated.  It's
	only partially specified, and designed to be extended.  Agent (game.py)
	is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent (question 2)
	"""

	def getAction(self, gameState):
		"""
		Returns the minimax action from the current gameState using self.depth
		and self.evaluationFunction.

		Here are some method calls that might be useful when implementing minimax.

		gameState.getLegalActions(agentIndex):
		Returns a list of legal actions for an agent
		agentIndex=0 means Pacman, ghosts are >= 1

		gameState.generateSuccessor(agentIndex, action):
		Returns the successor game state after an agent takes an action

		gameState.getNumAgents():
		Returns the total number of agents in the game

		gameState.isWin():
		Returns whether or not the game state is a winning state

		gameState.isLose():
		Returns whether or not the game state is a losing state
		"""
		"*** YOUR CODE HERE ***"
		value = float("-inf")
		bestAction = []
		agent_index = 0
		actions = gameState.getLegalActions(agent_index)
		successors = [(action, gameState.generateSuccessor(agent_index, action)) for action in actions]
		num_agents = gameState.getNumAgents()
		for successor in successors:
			result = minimax(1, range(num_agents), successor[1], self.depth, self.evaluationFunction)
			if result > value:
				value = result
				bestAction = successor[0]
		return bestAction


def minimax(agent, all_agents, state, depth, evalFunc):

	if depth <= 0 or state.isWin() == True or state.isLose() == True:
		return evalFunc(state)

	if agent == 0:
		stat = float("-inf")
	else:
		stat = float("inf")

	actions = state.getLegalActions(agent)
	successors = [state.generateSuccessor(agent, action) for action in actions]
	for i in range(len(successors)):
		successor = successors[i]
		if agent == 0:
			stat = max(stat, minimax(all_agents[agent + 1], all_agents, successor, depth, evalFunc))
		elif agent == all_agents[-1]:
			stat = min(stat, minimax(all_agents[0], all_agents, successor, depth - 1, evalFunc))
		else:
			stat = min(stat, minimax(all_agents[agent + 1], all_agents, successor, depth, evalFunc))

	return stat


class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"
		v = float("-inf")
		alpha = float("-inf")
		beta = float("inf")
		bestAction = []
		agent = 0
		actions = gameState.getLegalActions(agent)
		successors = [(action, gameState.generateSuccessor(agent, action)) for action in actions]
		for successor in successors:
			temp = minimaxPrune(1, range(gameState.getNumAgents()), successor[1], self.depth, self.evaluationFunction,
								alpha, beta)

			if temp > v:
				v = temp
				bestAction = successor[0]

			if v > beta:
				return bestAction

			alpha = max(alpha, v)

		return bestAction


def minimaxPrune(agent, agentList, state, depth, evalFunc, alpha, beta):
	if depth <= 0 or state.isWin() == True or state.isLose() == True:
		return evalFunc(state)

	if agent == 0:
		v = float("-inf")
	else:
		v = float("inf")

	actions = state.getLegalActions(agent)
	for action in actions:
		successor = state.generateSuccessor(agent, action)

		if agent == 0:
			v = max(v, minimaxPrune(agentList[agent + 1], agentList, successor, depth, evalFunc, alpha, beta))
			alpha = max(alpha, v)
			if v > beta:

				return v

		elif agent == agentList[-1]:
			v = min(v, minimaxPrune(agentList[0], agentList, successor, depth - 1, evalFunc, alpha, beta))
			beta = min(beta, v)
			if v < alpha:
				
				return v

		else:
			v = min(v, minimaxPrune(agentList[agent + 1], agentList, successor, depth, evalFunc, alpha, beta))
			beta = min(beta, v)
			if v < alpha:

				return v

	return v


class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	def getAction(self, gameState):
		"""
		Returns the expectimax action using self.depth and self.evaluationFunction

		All ghosts should be modeled as choosing uniformly at random from their
		legal moves.
		"""
		"*** YOUR CODE HERE ***"
		value = float("-inf")
		bestAction = []
		agent = 0
		actions = gameState.getLegalActions(agent)
		successors = [(action, gameState.generateSuccessor(agent, action)) for action in actions]
		for successor in successors:
			result = expectimax(1, range(gameState.getNumAgents()), successor[1], self.depth, self.evaluationFunction)
			if result > value:
				value = result
				bestAction = successor[0]
		return bestAction


def expectimax(agent, agentList, state, depth, evalFunc):

	if depth <= 0 or state.isWin() == True or state.isLose() == True:
		return evalFunc(state)

	if agent == 0:
		v = float("-inf")
	else:
		v = 0

	actions = state.getLegalActions(agent)
	successors = [state.generateSuccessor(agent, action) for action in actions]
	for successor in successors:
		if agent == 0:
			v = max(v, expectimax(agentList[agent + 1], agentList, successor, depth, evalFunc))
		elif agent == agentList[-1]:
			v = v + expectimax(agentList[0], agentList, successor, depth - 1, evalFunc)
		else:
			v = v + expectimax(agentList[agent + 1], agentList, successor, depth, evalFunc)

	if agent == 0:
		return v
	else:
		return v / float(len(successors))

def betterEvaluationFunction(currentGameState):
	"""
	Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	evaluation function (question 5).

	DESCRIPTION: <write something here so we know what you did>
	"""
	"*** YOUR CODE HERE ***"
	newPos = currentGameState.getPacmanPosition()
	newFood = currentGameState.getFood().asList()

	distance_to_food = float('inf')
	for food in newFood:
		d_food = manhattanDistance(newPos, food)
		distance_to_food = min(distance_to_food, d_food)

	d_ghost = 0
	for ghost in currentGameState.getGhostPositions():
		d_ghost = manhattanDistance(newPos, ghost)
		if (d_ghost < 2):
			return -float('inf')

	remaining_food = currentGameState.getNumFood()
	remaining_capsules = len(currentGameState.getCapsules())

	foodLeft_factor = 1000000
	capsLeft_factor = 10000
	foodDist_factor = 1000

	additionalFactors = 0
	if currentGameState.isLose():
		additionalFactors -= 50000
	elif currentGameState.isWin():
		additionalFactors += 50000

	return 1.0 / (remaining_food + 1) * foodLeft_factor + 1.0/ d_ghost + \
		   1.0 / (distance_to_food + 1) * foodDist_factor + \
		   1.0 / (remaining_capsules + 1) * capsLeft_factor + additionalFactors

# Abbreviation
better = betterEvaluationFunction

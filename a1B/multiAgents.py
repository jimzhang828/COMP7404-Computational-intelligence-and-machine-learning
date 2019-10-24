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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        prevFood = currentGameState.getFood()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0

        foodPositions = prevFood.asList()
        minDis = min([manhattanDistance(newPos, foodPos) for foodPos in foodPositions])
        score += 1.0 / (minDis + 1)

        ghostPositions = currentGameState.getGhostPositions()
        if len(ghostPositions) > 0:
            minDis = min([manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions])
            score -= 2.0 / (minDis + 1)

        return score
        

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
        def getMax(state, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return '', self.evaluationFunction(state)
            values = []
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value = getMin(successor, depth, 1)[1]
                values.append((action, value))
            return max(values, key=lambda x: x[1])

        def getMin(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return '', self.evaluationFunction(state)
            values = []
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    value = getMax(successor, depth + 1)[1]
                else:
                    value = getMin(successor, depth, agentIndex + 1)[1]
                values.append((action, value))
            return min(values, key=lambda x: x[1])

        return getMax(gameState, 0)[0]
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def getMax(state, depth, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return '', self.evaluationFunction(state)
            a = ''
            v = float("-inf")
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value = getMin(successor, depth, 1, alpha, beta)[1]
                v = max(v, value)
                if v == value: a = action
                if v > beta: return a, v
                alpha = max(alpha, v)
            return a, v

        def getMin(state, depth, agentIndex, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return '', self.evaluationFunction(state)
            a = ''
            v = float("inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    value = getMax(successor, depth + 1, alpha, beta)[1]
                else:
                    value = getMin(successor, depth, agentIndex + 1, alpha, beta)[1]
                v = min(v, value)
                if v == value: a = action
                if v < alpha: return a, v
                beta = min(beta, v)
            return a, v
        return getMax(gameState, 0, float('-inf'), float('inf'))[0]
        # util.raiseNotDefined()
        

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
        def getMax(state, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return '', self.evaluationFunction(state)
            values = []
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value = getMean(successor, depth, 1)
                values.append((action, value))
            return max(values, key=lambda x: x[1])

        def getMean(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            values = []
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    value = getMax(successor, depth + 1)[1]
                else:
                    value = getMean(successor, depth, agentIndex + 1)
                values.append(value)
            return sum(values) / 1.0 / len(values)

        return getMax(gameState, 0)[0]
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    position = currentGameState.getPacmanPosition()
    # foods
    foodPositions = currentGameState.getFood().asList()
    if len(foodPositions) == 0:
        return 10000
    distances = [manhattanDistance(position, food) for food in foodPositions]
    score += 10.0 / (min(distances) + 1)
    # capsules
    capsules = currentGameState.getCapsules()
    distances = [manhattanDistance(position, capsule) for capsule in capsules]
    if distances:
        score += 20.0 / (min(distances) + 1)
    # ghosts
    ghostStates = currentGameState.getGhostStates()
    # ghostPositions = currentGameState.getGhostPositions()
    distances = [manhattanDistance(position, ghost.getPosition()) for ghost in ghostStates]
    for ghost, distance in zip(ghostStates, distances):
        if ghost.scaredTimer > 0:
            score += 100.0 / (distance + 1)
        else:
            score -= 10.0 / (distance + 1)
    return score
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


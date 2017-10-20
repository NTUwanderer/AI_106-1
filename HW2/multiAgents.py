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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        finalWeight = 1000
        dangerousWeight = 800
        eatGhostWeight = 300
        expEatWeight = 200
        eatFoodWeight = 100
        distWeight = 1

        width = newFood.width
        height = newFood.height

        safeDis = min(min(5, width - 2), height - 2)
        maxLength = width + height - 4
        "*** YOUR CODE HERE ***"
        if successorGameState.isLose():
            return successorGameState.getScore() - finalWeight * maxLength

        if successorGameState.isWin():
            return successorGameState.getScore() + finalWeight * maxLength

        oldGhostStates = currentGameState.getGhostStates()
        oldScaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

        sumOfTimes = 0
        for time in newScaredTimes:
            sumOfTimes += time

        delta = 0

        if len(oldGhostStates) < len(newGhostStates):
            delta += (len(newScaredTimes) - len(oldScaredTimes)) * eatGhostWeight * maxLength
        else:
            for i in range(len(oldGhostStates)):
                dis = manhattanDistance(oldGhostStates[i].getPosition(), newGhostStates[i].getPosition())
                if oldScaredTimes[i] != 0 and dis > 1:
                    delta += eatGhostWeight * maxLength

        trueDis = [[maxLength for x in range(height)] for y in range(width)] 

        trueDis[newPos[0]][newPos[1]] = 0

        oldList = [newPos]
        newList = []
        
        counter = 1
        while len(oldList) > 0:
            for pos1 in oldList:
                cands = [(pos1[0]-1, pos1[1]), (pos1[0]+1, pos1[1]), (pos1[0], pos1[1]-1), (pos1[0], pos1[1]+1)]
                for cand in cands:
                    if not (successorGameState.hasWall(cand[0], cand[1])) and counter < trueDis[cand[0]][cand[1]]:
                        newList.append(cand)
                        trueDis[cand[0]][cand[1]] = counter


            counter += 1
            oldList = newList
            newList = []

        temp = 0

        for i in range(len(newGhostStates)):
            pos = newGhostStates[i].getPosition()
            x = int(pos[0])
            y = int(pos[1])
            dis = trueDis[x][y]
            if (newScaredTimes[i] >= 2 * dis):
                value = expEatWeight * (maxLength - dis)
                if (value >= temp):
                    temp = value
        delta += temp

        temp = maxLength
        for i in range(len(newGhostStates)):
            if newScaredTimes[i] > 0:
                continue
            pos = newGhostStates[i].getPosition()
            x = int(pos[0])
            y = int(pos[1])
            dis = trueDis[x][y]
            if (temp > dis):
                temp = dis
            
        if temp < safeDis:
            delta -= (safeDis - temp) * dangerousWeight * maxLength / safeDis

        if currentGameState.hasFood(newPos[0], newPos[1]):
            delta += eatFoodWeight * maxLength
        
        minDis = maxLength
        for food in newFood.asList():
            dis = trueDis[food[0]][food[1]]
            if (dis < minDis):
                minDis = dis

        delta += distWeight * (maxLength - dis)

        return successorGameState.getScore() + delta

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
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        def myGetAction(gameState, depth, agentIndex):
            if depth == 0:
                return (None, self.evaluationFunction(gameState))

            actions = gameState.getLegalActions(agentIndex)

            if agentIndex == gameState.getNumAgents() - 1:
                newIndex = 0
                newDepth = depth - 1
            else:
                newIndex = agentIndex + 1
                newDepth = depth

            extremePair = None
            isMax = (agentIndex == 0)
            for action in actions:
                newGameState = gameState.generateSuccessor(agentIndex, action)
                pair = myGetAction(newGameState, newDepth, newIndex)

                if extremePair == None or (isMax and pair[1] > extremePair[1]) or (not isMax and pair[1] < extremePair[1]):
                    extremePair = (action, pair[1])

            if extremePair == None:
                return (None, self.evaluationFunction(gameState))
            return extremePair

        return myGetAction(gameState, self.depth, 0)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        posInf = float('inf')
        negInf = -posInf

        def myGetAction(gameState, depth, agentIndex, alpha, beta):
            if depth == 0:
                return (None, self.evaluationFunction(gameState))

            actions = gameState.getLegalActions(agentIndex)

            if agentIndex == gameState.getNumAgents() - 1:
                newIndex = 0
                newDepth = depth - 1
            else:
                newIndex = agentIndex + 1
                newDepth = depth

            extremePair = None
            isMax = (agentIndex == 0)
            for action in actions:
                newGameState = gameState.generateSuccessor(agentIndex, action)
                pair = myGetAction(newGameState, newDepth, newIndex, alpha, beta)
                value = pair[1]

                if extremePair == None or (isMax and value > extremePair[1]) or (not isMax and value < extremePair[1]):
                    extremePair = (action, value)
                    if isMax and value > alpha:
                        alpha = value
                    elif not isMax and value < beta:
                        beta = value

                    if alpha >= beta:
                        break

            if extremePair == None:
                return (None, self.evaluationFunction(gameState))
            return extremePair

        return myGetAction(gameState, self.depth, 0, negInf, posInf)[0]

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
        def myGetAction(gameState, depth, agentIndex):
            if depth == 0:
                return (None, self.evaluationFunction(gameState))

            actions = gameState.getLegalActions(agentIndex)

            if agentIndex == gameState.getNumAgents() - 1:
                newIndex = 0
                newDepth = depth - 1
            else:
                newIndex = agentIndex + 1
                newDepth = depth

            totalValue = float(0.0)
            extremePair = None
            isMax = (agentIndex == 0)
            for action in actions:
                newGameState = gameState.generateSuccessor(agentIndex, action)
                pair = myGetAction(newGameState, newDepth, newIndex)

                if extremePair == None or (isMax and pair[1] > extremePair[1]) or (not isMax):
                    extremePair = (action, pair[1])
                    totalValue += pair[1]

            if extremePair == None:
                return (None, self.evaluationFunction(gameState))
            if not isMax:
                extremePair = (extremePair[0], totalValue / len(actions))

            return extremePair

        return myGetAction(gameState, self.depth, 0)[0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


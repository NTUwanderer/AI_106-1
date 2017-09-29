# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    start = problem.getStartState()
    nodes = Stack()
    indices = Stack()
    steps = []
    predecessors = []
    explored = set()
    frontier = set()
    nodes.push(start)
    indices.push(0)
    predecessors.append({'node': None, 'index': -1, 'dir': None})
    frontier.add(start)

    indexCounter = 1
    while nodes.isEmpty() == False:
        node = nodes.pop()
        index = indices.pop()
        explored.add(node)
        frontier.remove(node)

        if problem.isGoalState(node):
            while (predecessors[index].get('node') != None):
                predecessor = predecessors[index]
                node = predecessor.get('node')
                index = predecessor.get('index')
                steps.insert(0, predecessor.get('dir'))
            break

        successors = problem.getSuccessors(node)
        for successor in successors:
            if successor[0] not in explored and successor[0] not in frontier:
                newIndex = indexCounter
                indexCounter += 1
                nodes.push(successor[0])
                indices.push(newIndex)
                frontier.add(successor[0])
                predecessors.append({'node': node, 'index': index, 'dir': successor[1]})


    return steps
    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    start = problem.getStartState()
    nodes = Queue()
    indices = Queue()
    steps = []
    predecessors = []
    explored = set()
    frontier = set()
    nodes.push(start)
    indices.push(0)
    predecessors.append({'node': None, 'index': -1, 'dir': None})
    frontier.add(start)

    indexCounter = 1
    while nodes.isEmpty() == False:
        node = nodes.pop()
        index = indices.pop()
        explored.add(node)
        frontier.remove(node)

        if problem.isGoalState(node):
            while (predecessors[index].get('node') != None):
                predecessor = predecessors[index]
                node = predecessor.get('node')
                index = predecessor.get('index')
                steps.insert(0, predecessor.get('dir'))
            break

        successors = problem.getSuccessors(node)
        for successor in successors:
            if successor[0] not in explored and successor[0] not in frontier:
                newIndex = indexCounter
                indexCounter += 1
                nodes.push(successor[0])
                indices.push(newIndex)
                frontier.add(successor[0])
                predecessors.append({'node': node, 'index': index, 'dir': successor[1]})


    return steps

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    start = problem.getStartState()
    nodes = PriorityQueue()
    indices = PriorityQueue()
    steps = []
    predecessors = []
    costs = []
    explored = set()
    frontier = set()
    nodes.push(start, 0)
    indices.push(0, 0)
    predecessors.append({'node': None, 'index': -1, 'dir': None})
    costs.append(0)
    frontierIndices = {}
    frontierIndices[start] = 0

    indexCounter = 1
    while nodes.isEmpty() == False:
        node = nodes.pop()
        index = indices.pop()
        cost = costs[index]

        del frontierIndices[node]
        explored.add(node)

        if problem.isGoalState(node):
            while (predecessors[index].get('node') != None):
                predecessor = predecessors[index]
                node = predecessor.get('node')
                index = predecessor.get('index')
                steps.insert(0, predecessor.get('dir'))
            break

        successors = problem.getSuccessors(node)
        for successor in successors:
            if successor[0] not in explored:
                newCost = cost + successor[2]
                if successor[0] in frontierIndices:
                    frontierIndex = frontierIndices[successor[0]]
                    oldCost = costs[frontierIndex]
                    if (newCost < oldCost):
                        nodes.update(successor[0], newCost)
                        indices.update(frontierIndex, newCost)
                        predecessors[frontierIndex] = {'node': node, 'index': index, 'dir': successor[1]}
                        costs[frontierIndex] = newCost
                else:
                    newIndex = indexCounter
                    indexCounter += 1
                    nodes.push(successor[0], newCost)
                    indices.push(newIndex, newCost)
                    frontierIndices[successor[0]] = newIndex
                    predecessors.append({'node': node, 'index': index, 'dir': successor[1]})
                    costs.append(newCost)

    return steps

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

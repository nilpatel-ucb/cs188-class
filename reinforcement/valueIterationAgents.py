# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #Transition fxn (using state, action, and next state)
        #Reward fxn (of going to next state using action, from current state)
        #discounted old guess of the future (gamma*Vk)
        #Return max(Transition fxn * [Reward fxn + Discounted old guess)]
        for iter in range(self.iterations):
            VkPlus1Values = util.Counter()
            for state in self.mdp.getStates():
              
                if self.mdp.isTerminal(state):
                    VkPlus1Values[state] = 0
                    continue
                #no actions to take for a state then jus move on
                if len(self.mdp.getPossibleActions(state)) == 0:
                    continue
                maxQVal = float('-inf')
                
                for  action in  self.mdp.getPossibleActions(state):
                    Q = 0

                    for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        #self.values holds the old values that we need to compare against
                        discountedOldGuess = self.discount * self.values[nextState]

                        rewardFxn =  self.mdp.getReward(state, action, nextState)
                        #probabity is the transition fxn probalbity

                        # for action in self.mdp.getPossibleActions(state):
                        #     for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                #         reward = self.mdp.getReward(state, action, nextState)
                #         discountedOldGuess = self.discount * self.values(nextState)
                    #[(nextState, Prob) (East, .3),(West, .2)] )
            

                        Q += probability * (rewardFxn + discountedOldGuess)
                    maxQVal = max(maxQVal, Q)
                VkPlus1Values[state] = maxQVal

            self.values = VkPlus1Values

        return

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
      
        Q = 0

        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                #self.values holds the old values that we need to compare against
                discountedOldGuess = self.discount * self.values[nextState]

                rewardFxn =  self.mdp.getReward(state, action, nextState)
                        #probabity is the transition fxn probalbity

                        # for action in self.mdp.getPossibleActions(state):
                        #     for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                #         reward = self.mdp.getReward(state, action, nextState)
                #         discountedOldGuess = self.discount * self.values(nextState)
                    #[(nextState, Prob) (East, .3),(West, .2)] )
            

                Q += probability * (rewardFxn + discountedOldGuess)
        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        policy = None

        if self.mdp.isTerminal(state):
            return None
        if len(self.mdp.getPossibleActions(state)) == 0:
            return None

        maxValue = float('-inf')

        for action in self.mdp.getPossibleActions(state):
            Q = self.computeQValueFromValues(state,action)

            if Q  > maxValue:
                policy = action 
                maxValue = Q
        return policy



        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"


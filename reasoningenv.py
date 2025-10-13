import gymnasium as gym
import basictypes
import knowledgebase
import numpy as np
import reasoner
import kbparser
import csv
import kbencoder
from copy import deepcopy
from typing import Callable
from itertools import count


class Node():
    def __init__(self, data, parent):
        self.parent = parent
        self.data = data
        if self.parent is not None:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    def __str__(self):
        return str(self.parent)


class BackwardChainEnv(gym.Env):
    # self.kb: knowledgebase object
    # self.embedded_kb: numpy array of the embeddings of the kb rules
    # self.queries: knowledgebase object
    # self.embedded_q: embeddings of query kb rules
    # self.state: Node of rule and parent node
    # self.embed: function that generates an embedding from either an atom or rule
    #
    metadata = {"render_modes": ["ansi"]}

    def __init__(self,
                 kb: knowledgebase.KnowledgeBase,
                 queries: knowledgebase.KnowledgeBase,
                 embed: Callable,
                 render_mode=None,
                 max_depth: int = 5):
        self.max_depth = max_depth
        self.embed = embed

        self.kb = kb
        self.embedded_kb = np.array([self.embed(i) for i in self.kb.rules])  # Numpy array

        self.queries = deepcopy(queries)
        self.embedded_q = [self.embed(i) for i in self.queries.rules]

        self.observation_space = gym.spaces.Dict(
            {
                "query": gym.spaces.Box(low=-np.inf, high=np.inf,
                                        dtype=np.float32,
                                        shape=self.embedded_kb[0].shape),
                "rules": gym.spaces.Box(low=-np.inf, high=np.inf,
                                        dtype=np.float32,
                                        shape=self.embedded_kb.shape)
            }
        )

        # An action should be the index of the rule and the index of the goal
        # atom
        self.action_space = gym.spaces.Box(low=np.array([0, 0]),
                                           high=np.array([len(self.kb.rules),
                                                          0]),
                                           dtype=np.int32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {
            "query": self.embed(self.state.data),
            "rules": self.embedded_kb
        }

    def _get_info(self):
        return {
            "query": self.state.data,
            "rules": self.kb
        }

    def reset(self, seed=None, options={"new_query": True}):
        super().reset(seed=seed, options=options)
        if options["new_query"]:
            self.query = self.queries.rules.pop().head

        var_count = 0
        variables = []
        for i in self.query.arguments:
            if isinstance(i, basictypes.Variable):
                var_count += 1
                variables.append(i)

        query_clause = knowledgebase.Rule(basictypes.Atom(basictypes.Predicate(var_count, "yes"),
                                                          variables),
                                          [self.query])
        self.state = Node(query_clause, None)
        self.action_space = gym.spaces.Box(low=np.array([0, 0]),
                                           high=np.array([len(self.kb.rules)-1,
                                                          len(self.state.data.body)-1]),
                                           dtype=np.int32)
        self.actions = self.get_actions()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "ansi":
            print(self.state.data)

        return observation, info

    def step(self, action):
        rule = deepcopy(self.kb.rules[action[0]])

        # Replace the variable names in the chosen rule.
        # We must use the replaced names' rule as the first argument
        # in the call to reasoner.unify(), since then the replaced names
        # will all be substituted away.
        reasoner.standardize(rule, self.state.depth)

        substitution = reasoner.unify(rule.head,
                                      self.state.data.body[action[1]])

        if substitution is not False:
            if self.render_mode == "ansi":
                print(f"Answer clause: {self.state.data}")
                print(f"Rule: {rule}")

            query_clause = deepcopy(self.state.data)
            # Complete the SLD resolution
            del query_clause.body[action[1]]
            if rule.body:
                for i in rule.body:
                    query_clause.body.append(kbencoder.clean_atom(i))

            query_clause = reasoner.sub_rule(query_clause, substitution)

            self.state = Node(query_clause, self.state)
            self.action_space = gym.spaces.Box(low=np.array([0, 0]),
                                               high=np.array([len(self.kb.rules)-1,
                                                              max(0, len(self.state.data.body)-1)]),
                                               dtype=np.int32)
            self.actions, terminated, reward = self.get_action_terminal()

            # We're done modifying the query, so we can go ahead
            # and get the observation and info
            observation = self._get_obs()
            info = self._get_info()

            # Print out the current query state if we are in a terminal state,
            # since then step() will not be called again.
            if terminated and self.render_mode == "ansi":
                print(f"Answer clause: {self.state.data}")

            if self.state.depth <= self.max_depth:
                truncated = False
            else:
                truncated = True

            return observation, reward, terminated, truncated, info
        else:
            reward = -1
            observation = self._get_obs()
            info = self._get_info()

            if self.render_mode == "ansi":
                print(f"Rule {rule} and query body atom {self.state.data.body[action[1]]} did not unify.")

            return observation, reward, False, False, info

    def terminal_condition(self):
        # Determine whether or not we are in a terminal state
        # In such a state, we have either found an answer, or
        # we know we cannot find an answer.

        terminated = False
        reward = 0
        # Check whether the query is now an answer
        if not self.state.data.body:
            terminated = True
            reward = 100
        else:
            # Check whether all query atoms have something they can unify with.
            # If that's not the case, we know that we can't find an answer.
            for atom in self.state.data.body:
                terminated = True
                for rule in self.kb.rules:
                    if reasoner.unify(atom, rule.head) is not False:
                        terminated = False
                if terminated is True:
                    break
        return terminated, reward

    def action_embedding(self, action: []):
        '''Return the embedding of an action.'''
        rule = self.kb.rules[action[0]]
        goal = self.state.data.body[action[1]]
        embedding = np.concatenate((self.embed(goal), self.embed(rule)))
        return embedding

    def get_actions(self):
        '''Returns a list of tuples of actions and their embeddings.
        The actions are themselves tuples, whereas the embeddings are numpy
        arrays.'''
        actions = []
        # For every combination of a rule and query body atom
        for j in range(self.action_space.high[1]+1):
            for i in range(self.action_space.high[0]+1):
                rule = self.kb.rules[i]
                goal = self.state.data.body[j]
                if reasoner.unify(goal, rule.head) is not False:
                    action = (i, j)
                    embedding = np.concatenate((self.embed(goal), self.embed(rule)))
                    actions.append((action, embedding))
        return actions

    def get_action_terminal(self):
        '''Returns actions, terminated, reward'''
        actions = []
        if self.state.data.body:
            for i in range(self.action_space.high[1]+1):
                terminated = True
                for j in range(self.action_space.high[0]+1):
                    goal = self.state.data.body[i]
                    rule = self.kb.rules[j]
                    if reasoner.unify(goal, rule.head) is not False:
                        terminated = False
                        action = (j, i)
                        embedding = np.concatenate((self.embed(goal),
                                                    self.embed(rule)))
                        actions.append((action, embedding))
                if terminated:
                    return [], True, 0
            return actions, False, 0
        else:
            return [], True, 100.

    def backtrack(self):
        self.state = self.state.parent
        self.action_space = gym.spaces.Box(low=np.array([0, 0]),
                                           high=np.array([len(self.kb.rules)-1,
                                                          len(self.state.data.body)-1]),
                                           dtype=np.int32)


def select_action_standard(env, observation, reward, terminated, truncated,
                           info, additional):
    return env.actions


def backwardchain(env: BackwardChainEnv, select_action: Callable,
                  steps: count, additional: list):
    def backwardchainaction(env: BackwardChainEnv, action):
        observation, reward, terminated, truncated, info = env.step(action)
        next(steps)
        if not (terminated or truncated):
            actions = select_action(env, observation, reward, terminated,
                                    truncated, info, additional)
            if actions:
                for i in range(len(actions)):
                    next_action = actions[i][0]
                    if backwardchainaction(env, next_action):
                        return True
                env.backtrack()
                return False
            else:
                print("This should never happen?")
                env.backtrack()
                return False
        elif truncated:
            env.backtrack()
            return False
        elif terminated:
            if reward > 0:
                return True
            else:
                env.backtrack()
                return False

    observation, info = env.reset(options={"new_query": True})
    print(env.query)

    actions = select_action(env, observation, 0, False, False,
                            info, additional)
    if actions:
        for i in range(len(actions)):
            next_action = actions[i][0]
            if backwardchainaction(env, next_action):
                return True
        return False
    else:
        return False


def standardreasoner(kb: knowledgebase.KnowledgeBase,
                     queries: knowledgebase.KnowledgeBase,
                     max_depth: int = 5,
                     verbose: int = 1,
                     csvpath: str = None):
    def embed(rule):
        return np.array([])

    if verbose <= 1:
        env = BackwardChainEnv(kb, queries, embed, None, max_depth)
    else:
        env = BackwardChainEnv(kb, queries, embed, "ansi", max_depth)

    if csvpath:
        with open(csvpath, "a") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Query", "Depth", "Steps", "Answer", "Failed"])

    for i in range(len(queries.rules)):
        steps = count()

        backwardchain(env, select_action_standard, steps, [])

        if csvpath:
            with open(csvpath, "a") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([str(env.query),
                                    env.state.depth,
                                    next(steps),
                                    str(env.state.data),
                                    True if env.state.data.body else False])
        steps = count()


# Test the environment by implementing the standard reasoner
if __name__ == "__main__":
    def embed(rule):
        return np.array([])

    kb = kbparser.parse_KB_file("randomKB.txt")
    queries = kbparser.parse_KB_file("test_queries.txt")

    standardreasoner(kb, queries, 1, "rl_standard_data.csv")

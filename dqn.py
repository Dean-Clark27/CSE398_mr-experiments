#
# Largely lifted from pytorch's DQN tutorial
# Although not so much, now.
# TODO: Document all of the functions that I've written
#
import reasoningenv
import torch
import random
import argparse
import reasoner
import kbparser
import knowledgebase
import chainbased
import math
import gymnasium as gym
import numpy as np
import csv
from collections import namedtuple, deque
from itertools import count
from copy import deepcopy
from typing import Callable


Transition = namedtuple("Transition",
                        ("state", "action", "next_state",
                         "next_state_action", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, a: int):
        return self.memory[a]

    def __setitem__(self, a: int, b):
        self.memory[a] = b


class DQN(torch.nn.Module):
    def __init__(self, input_size):
        super(DQN, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 128)
        self.rel1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(128, 128)
        self.rel2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = self.rel1(self.l1(x))
        x = self.rel2(self.l2(x))
        return self.l3(x)


def select_action(state, info, policy_net, env,
                  eps_start, eps_end, eps_decay, device):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1

    actions = env.actions
    if sample > eps_threshold:
        action_array = []
        with torch.no_grad():
            for i in actions:
                input_tensor = torch.cat((torch.tensor(i[1], device=device),
                                          state[0]))
                action_value = policy_net(input_tensor).item()
                action_array.append([i[0], i[1], action_value])
            action_array.sort(key=lambda x: x[-1], reverse=True)
            action = action_array[0]
            del action[-1]  # We do not need the score to be returned
            return action
    else:
        action = random.choice(actions)
        return action


def select_action_dqn(env, observation, reward, terminated, truncated, info,
                      additional: list):
    '''The additional argument should have the model and the device,
    in that order.'''
    policy_net = additional[0]
    device = additional[1]
    state = torch.tensor(observation, dtype=torch.float32,
                         device=device).unsqueeze(0)
    action_array = []
    actions = env.actions
    with torch.no_grad():
        for i in actions:
            input_tensor = torch.cat((torch.tensor(i[1], device=device),
                                      state[0]))
            action_value = policy_net(input_tensor).item()
            action_array.append([i[0], action_value])
        action_array.sort(key=lambda x: x[-1], reverse=True)
        return action_array


# Note that the reasoner just goes in order through the array, so we just sort
# differently
def select_action_dqn_lowest(env, observation, reward,
                             terminated, truncated, info, additional):
    '''The additional argument should have the model and the device,
    in that order.'''
    policy_net = additional[0]
    device = additional[1]
    state = torch.tensor(observation, dtype=torch.float32,
                         device=device).unsqueeze(0)
    action_array = []
    action_dict = {}
    actions = env.actions
    with torch.no_grad():
        for i in actions:
            input_tensor = torch.cat((torch.tensor(i[1], device=device),
                                      state[0]))
            action_value = policy_net(input_tensor).item()
            if i[0][1] in action_dict:
                action_dict[i[0][1]].append((i[0], action_value))
            else:
                action_dict[i[0][1]] = [(i[0], action_value)]
        goal_order = []
        for goal in action_dict:
            action_dict[goal].sort(key=lambda x: x[-1], reverse=True)
            goal_order.append((goal, action_dict[goal][0][-1]))
#        print(goal_order)
        goal_order.sort(key=lambda x: x[-1])
#        print(goal_order)
        for i in goal_order:
            action_array = action_array + action_dict[i[0]]
#        print(action_array)
        return action_array


def optimize_model(memory, batch_size, device, gamma,
                   policy_net, target_net, env, optimizer,
                   verbose: int = 1):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)

    batch = Transition(*zip(*transitions))

    # Gets the next states as long as there is one
    non_final_mask = torch.tensor(tuple(map(lambda s, y: s is not None and y is not None,
                                            batch.next_state, batch.next_state_action)),
                                  device=device,
                                  dtype=torch.bool)

    non_final_next_states = torch.cat([y for s, y in zip(batch.next_state_action, batch.next_state)
                                       if s is not None and y is not None])
    non_final_next_state_actions = torch.cat([s for s, y in zip(batch.next_state_action, batch.next_state)
                                              if s is not None and y is not None])
    non_final_next_states = torch.cat((non_final_next_state_actions,
                                       non_final_next_states),
                                      dim=1)

    action_batch = torch.cat(batch.action)
    state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)

    input_batch = torch.cat((action_batch, state_batch), dim=1)

    state_action_values = policy_net(input_batch)

    # I think we don't even have to worry about this part. All it does is get
    # the score of the action that was actually used, but since we already have
    # that we need not worry about it. Nevertheless, I leave this here as a
    # record just in case I've messed something up.
    # state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).squeeze(1)
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()


def train_model(policy_net: DQN, kb: knowledgebase.KnowledgeBase,
                train_queries: knowledgebase.KnowledgeBase,
                embed: Callable, device: str, max_depth: int,
                learning_rate: float, iterations: int,
                epsilon_start: float, epsilon_end: float, epsilon_decay: int,
                batch_size: int, gamma: float, tau: float, verbose: int = 1):
    target_net = deepcopy(policy_net)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.AdamW(policy_net.parameters(),
                                  lr=learning_rate, amsgrad=True)
    memory = ReplayMemory(50000)

    global steps_done
    steps_done = 0

    episode_durations = []

    env = reasoningenv.BackwardChainEnv(kb, train_queries, embed, None, max_depth)
    env = gym.wrappers.FlattenObservation(env)

    for i_query in range(len(env.queries.rules)):
        if verbose >= 1:
            print(f"{i_query} queries completed/{len(env.queries.rules)} remaining")
        state, info = env.reset(options={"new_query": True})

        for i in range(iterations):
            state = torch.tensor(state, dtype=torch.float32,
                                 device=device).unsqueeze(0)
            for t in count():
                action = select_action(state,
                                       info,
                                       policy_net,
                                       env,
                                       epsilon_start,
                                       epsilon_end,
                                       epsilon_decay,
                                       device)
                if t > 0:
                    a = memory[-1]
                    memory[-1] = Transition(a.state, a.action, a.next_state,
                                            torch.tensor(action[1],
                                                         device=device).unsqueeze(0),
                                            a.reward)

                # The returned action is an ordered pair of the indices,
                # which we use as an argument to env.step(), and the
                # embedding, which we push to the memory.
                observation, reward, terminated, truncated, info = env.step(action[0])
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation,
                                              dtype=torch.float32,
                                              device=device).unsqueeze(0)

                # Since the embedding is a numpy array, we convert it to a
                # tensor
                if not truncated:
                    memory.push(state,
                                torch.tensor(action[1],
                                             device=device).unsqueeze(0),
                                next_state,
                                None,
                                reward)

                state = next_state

                loss = optimize_model(memory, batch_size, device, gamma,
                                      policy_net, target_net, env, optimizer)
                if verbose >= 2:
                    print(loss)

                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = target_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * tau \
                                                    + target_net_state_dict[key] * (1 - tau)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    break
            state, info = env.reset(options={"new_query": False})


def generate_training_queries(facts: knowledgebase.KnowledgeBase):
    queries = []
    for i in range(args.queries + 1):
        query = knowledgebase.Rule(reasoner.gen_random_query(facts.rules),
                                   [])
        queries.append(query)
    queries = knowledgebase.KnowledgeBase(queries)
    return queries


def generate_testing_queries(train_queries: knowledgebase.KnowledgeBase,
                             num_queries: int = 100):
    test_queries = []
    for i in range(num_queries):
        query = knowledgebase.Rule(reasoner.gen_random_query(facts.rules),
                                   [])
        while query in train_queries.rules:
            query = knowledgebase.Rule(reasoner.gen_random_query(facts.rules),
                                       [])
        test_queries.append(query)
    test_queries = knowledgebase.KnowledgeBase(test_queries)
    return test_queries


def get_input_size(kb, queries, embed):
    env = reasoningenv.BackwardChainEnv(kb, queries, embed, None)
    env = gym.wrappers.FlattenObservation(env)
    state, info = env.reset(options={"new_query": True})
    actions = env.actions
    while not actions:
        state, info = env.reset(options={"new_query": True})
        actions = env.actions
    action_embed_size = len(actions[0][1])
    n_observations = len(state)
    input_size = action_embed_size + n_observations
    return input_size


# TODO: Should probably actually be moved to reasoningenv.py
def guidedreasoner(kb: knowledgebase.KnowledgeBase,
                   queries: knowledgebase.KnowledgeBase,
                   policy_net: DQN,
                   embed: Callable,
                   select: Callable,
                   max_depth: int = 5,
                   verbose: int = 1,
                   csvpath: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose <= 1:
        env = reasoningenv.BackwardChainEnv(kb, queries, embed, None, max_depth)
    else:
        env = reasoningenv.BackwardChainEnv(kb, queries, embed, "ansi", max_depth)
    env = gym.wrappers.FlattenObservation(env)

    if csvpath:
        with open(csvpath, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Query", "Depth", "Steps", "Answer", "Failed"])

    # Test the performance of the model on the generated queries
    for i in range(len(queries.rules)):
        steps = count()

        reasoningenv.backwardchain(env, select, steps, [policy_net, device])

        if csvpath:
            with open(csvpath, "a") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([str(env.query),
                                    env.state.depth,
                                    next(steps),
                                    str(env.state.data),
                                    True if env.state.data.body else False])
        steps = count()


def guidedreasoner_nobacktrack(kb: knowledgebase.KnowledgeBase,
                               queries: knowledgebase.KnowledgeBase,
                               policy_net: DQN,
                               embed: Callable,
                               select: Callable,
                               verbose: int = 1,
                               csvpath: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose <= 1:
        env = reasoningenv.BackwardChainEnv(kb, queries, embed, None)
    else:
        env = reasoningenv.BackwardChainEnv(kb, queries, embed, "ansi")
    env = gym.wrappers.FlattenObservation(env)

    if csvpath:
        with open(csvpath, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Query", "Depth", "Steps", "Answer", "Failed"])

    # Test the performance of the model on the generated queries
    reward_list = []
    depth_list = []
    for i in range(len(queries.rules)):
        observation, info = env.reset(options={"new_query": True})
        state = torch.tensor(observation, dtype=torch.float32,
                             device=device).unsqueeze(0)
        terminated, truncated, reward = False, False, 0
        for t in count():
#            action = select_action(state, info, policy_net, env, 0., 0., 1., device)
            action = select_action_dqn(env, observation, reward, terminated, truncated, info, [policy_net, device])[0]
#            action = env.actions[0]
#            action = select_action_test(env, observation, reward, terminated, truncated, info, [policy_net, device])
            observation, reward, terminated, truncated, info = env.step(action[0])
            state = torch.tensor(observation, dtype=torch.float32,
                                 device=device).unsqueeze(0)
            done = terminated or truncated

            if done:
                print(env.query, reward)
                reward_list.append(reward)
                if reward > 0:
                    depth_list.append(env.state.depth)
                break
    print(sum(reward_list)/100)
    print(max(depth_list))


if __name__ == "__main__":
    # Define the embedding functions
    # TODO: Define additional embedding functions and add CLI arguments to
    # control their use
    def chainbased_embed(rule):
        return chainbased.represent_pattern(rule, 10)

    aparser = argparse.ArgumentParser(description="Run the program with \
    defaults and all options on: python dqn.py -t -r -s rl-policy.pth -g 100 \
    -e ; Load an existing model, generate queries, and \
    run the experiment: python dqn.py -l rl-policy.pth -g 100 -e")
    aparser.add_argument("-k", "--knowledge_base", default="randomKB.txt",
                         help="Path to the knowledge base. Default: randomKB.txt")
    aparser.add_argument("-f", "--facts_list", default="random_facts.txt",
                         help="Path to the list of facts. Default: random_facts.txt")
    aparser.add_argument("-t", "--train", action="store_true",
                         help="Train the model.")
    aparser.add_argument("-r", "--generate_train_queries", action="store_true",
                         help="Generate new training queries instead of using \
                         the queries from the training queries file path.")
    aparser.add_argument("-q", "--queries", type=int, default=200,
                         help="Number of queries to generate for training. \
                         Default: 200")
    aparser.add_argument("-i", "--iterations", type=int, default=20,
                         help="Number of times to train per query. Default: 20")
    aparser.add_argument("-s", "--save_model",
                         help="Path to save the model.")
    aparser.add_argument("-l", "--load_model",
                         help="Path to load the model.")
    aparser.add_argument("-g", "--generate_test_queries", type=int,
                         help="Generate new test queries. This switch should \
                         be on if generating new training queries, since we \
                         don't want to repeat any queries.")
    aparser.add_argument("-e", "--test", action="store_true",
                         help="Test the trained or loaded model on the test \
                         queries.")
    aparser.add_argument("-v", "--verbose", action="store_true",
                         help="Print out what the reasoner is doing")
    aparser.add_argument("--train_query_path", default="train_queries.txt",
                         help="Path to save the list of queries used for \
                         training. Default: train_queries.txt")
    aparser.add_argument("--test_query_path", default="test_queries.txt",
                         help="Path to save the list of queries used to test \
                         the model. Default: test_queries.txt")
    aparser.add_argument("--unification_model", default="rKB_model.pth",
                         help="Path to the unification embedding model. \
                         Default: rKB_model.pth")
    aparser.add_argument("--batch_size", type=int, default=128,
                         help="Default: 128")
    aparser.add_argument("--gamma", type=float, default=0.99,
                         help="Default: 0.99")
    aparser.add_argument("--epsilon_start", type=float, default=0.9,
                         help="Default: 0.9")
    aparser.add_argument("--epsilon_end", type=float, default=0.05,
                         help="Default: 0.05")
    aparser.add_argument("--epsilon_decay", type=int, default=2000,
                         help="Default: 2000")
    aparser.add_argument("--tau", type=float, default=0.005,
                         help="Default: 0.005")
    aparser.add_argument("--learning_rate", type=float, default=1e-4,
                         help="Default: 1e-4")
    args = aparser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device} device")

    # Parse the knowledge base and the list of all facts that can be inferred
    # from it
    kb = kbparser.parse_KB_file(args.knowledge_base)
    facts = kbparser.parse_KB_file(args.facts_list)

    # Generate the queries to be used for training
    if args.generate_train_queries:
        queries = generate_training_queries(facts)
        kbparser.KB_to_txt(queries, args.train_query_path)
    queries = kbparser.parse_KB_file(args.train_query_path)

    # Choose the embedding function
    embed = chainbased_embed

    # Get data from the environment
    env = reasoningenv.BackwardChainEnv(kb, queries, embed, None)
    env = gym.wrappers.FlattenObservation(env)

    state, info = env.reset(options={"new_query": True})
    action_embed_size = len(env.actions[0][1])
    n_observations = len(state)

    input_size = n_observations + action_embed_size

    # Set up the model
    policy_net = DQN(input_size).to(device)

    # Reinitialize the environment so that we use all of the training queries
    if args.verbose:
        env = reasoningenv.BackwardChainEnv(kb, queries, embed, "ansi")
    else:
        env = reasoningenv.BackwardChainEnv(kb, queries, embed, None)
    env = gym.wrappers.FlattenObservation(env)

    # Train the model
    if args.train:
        train_model(policy_net, kb, queries, embed, device,
                    args.learning_rate, args.iterations,
                    args.epsilon_start, args.epsilon_end, args.epsilon_decay,
                    args.batch_size, args.gamma, args.tau, 1)

    # Save policy_net and target_net to the given file paths if the relevant
    # switch is on
    if args.save_model is not None:
        torch.save(policy_net.state_dict(), args.save_model)

    # We can also load policy_net if we're just running the experiment without
    # training, etc.
    if args.load_model is not None:
        policy_net.load_state_dict(torch.load(args.load_model,
                                              map_location=torch.device(device)))

    # Generate one hundred new queries for testing. This should avoid any
    # duplicates of the queries used for training. The queries will be saved to
    # disk.
    if args.generate_test_queries:
        test_queries = generate_testing_queries(queries,
                                                args.generate_test_queries)
        kbparser.KB_to_txt(test_queries, args.test_query_path)

    # Test the model using the queries at the given path. If the queries were
    # generated, this will read in the queries.
    if args.test:
        test_queries = kbparser.parse_KB_file(args.test_query_path)
        policy_net.eval()
        global steps_done
        steps_done = 0
        guidedreasoner_nobacktrack(kb, test_queries, policy_net, embed, select_action_dqn,
                                   args.verbose, "rl_guided_data.csv")

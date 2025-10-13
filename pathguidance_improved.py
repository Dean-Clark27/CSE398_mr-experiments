# **** This class is deprecated *****
# All useful functionality has been moved to mr_back_reasoner.py and evaluate.py
# The score_rule_query_{termwalk | chainbased) functions are broken due to the recent refactoring of
# related functions.
from functools import lru_cache
import sys
import time
import os
import csv
from typing import Literal, Union
import chainbased
from prints import clear_line, print_progress_bar
import termwalk
import argparse
import autoencoder
import kbencoder
import nnreasoner
from time import process_time
from basictypes import Atom, Predicate, Variable
from copy import copy
import nnunifier
import kbparser
import numpy as np
from kbparser import parse_KB_file
import knowledgebase
import reasoner
import torch

from vocab import Vocabulary

# need this to fix interrupt issues, must be before any scipy is imported
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

# type aliases
reasoner_name_type = Union[str, Literal["unity"],
                           Literal["autoencoder"], Literal["chainbased"], Literal["termwalk"]]
rule_type = tuple[knowledgebase.Rule, float, Atom, dict | bool]


# When this option is used, a rule must have at least the score to be evaluated
MIN_RULE_SCORE = 0.01
# MIN_RULE_SCORE = 0.001
# FALLBACK_DEPTH = 3  # The depth at which the meta-reasoner switches to the standard reasoner
FALLBACK_DEPTH = 5
NODE_MAX = 10_000_000
global DEBUG

# The maximum number of reasoning steps that trace information will be output for (per query)
TRACE_MAX = 150
TRACE_UP_TO_MIN = 2  # Level of search shown by trace, even after max nodes is reached

timeToExecute = 0

vocab = Vocabulary()
cache = reasoner.CachedUnify()


class GoalRuleCache:
    @lru_cache(maxsize=None)
    def generate_goals(self, arg: Atom, KB: knowledgebase.KnowledgeBase, depth: int):
        valid_rules: list[rule_type] = []
        max_score = 0.0

        for rule in KB.rule_by_pred[arg.predicate]:
            rule_1: knowledgebase.Rule = copy(rule)
            reasoner.standardize(rule_1, depth)
            subst = cache.unify_memoized(rule_1.head, arg)
            # subst = reasoner.old_unify(rule_1.head, arg)

            if not isinstance(subst, dict):
                continue

            score = score_rule_query(arg, rule, model, guidance_model)

            if not score:
                continue

            addition = (rule_1, score, arg, subst)
            valid_rules.append(addition)
            max_score = max(max_score, score)

        valid_rules.sort(key=lambda x: x[1], reverse=True)

        max_score = 1 if len(valid_rules) < 1 else max_score

        return valid_rules, max_score


grc = GoalRuleCache()


def score_rule_query(query, rule, model, guidance_model: nnreasoner.NeuralNet) -> float:
    """Evaluates query and rule and returns a score.

    :param query: A subgoal (an atom) to evaluate
    :param rule: A rule that could be used to prove the goal
    :param model: The embedding model
    :param guidance_model: The reasoning model
    :return: A score (>=0, <=1) of the likelihood that the rule will eventually lead to a proof
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        query = {query: rule}
        query = kbencoder.one_hot_encode_query(query, vocab)

        # Extract tensors outside the loop
        atom = model(torch.FloatTensor(query[0]).to(
            device)).cpu().detach().numpy()
        rule_head = model(torch.FloatTensor(
            query[1]).to(device)).cpu().detach().numpy()
        args = torch.zeros(embed_size).to(device)

        for arg in query[2]:
            arg = model(torch.FloatTensor(arg).to(device))
            args += arg

        args = args.cpu().detach().numpy()

        # Convert NumPy arrays to PyTorch tensors
        atom = torch.FloatTensor(atom)
        rule_head = torch.FloatTensor(rule_head)
        args = torch.FloatTensor(args)

        # Concatenate tensors using torch.cat
        embedding = torch.cat([atom, rule_head, args])

        score = nnreasoner.get_score(embedding.numpy(), guidance_model)
        return score


def score_rule_query_termwalk(
    query,
    rule,
    guidance_model,
    num_pred: int = 10,
    num_var: int = 10,
    num_const: int = 100,
):
    with torch.no_grad():
        query_vec = termwalk.termwalk_representation(
            query, num_pred, num_var, num_const
        )
        rule_vec = termwalk.termwalk_representation(
            rule, num_pred, num_var, num_const)
        embedding = np.concatenate([query_vec, rule_vec])
        score = nnreasoner.get_score(embedding, guidance_model)
        return score


def score_rule_query_chainbased(query, rule, guidance_model):
    with torch.no_grad():
        query_vec = chainbased.represent_pattern(query, 20)
        rule_vec = chainbased.represent_pattern(rule, 20)
        embedding = np.concatenate([query_vec, rule_vec])
        score = nnreasoner.get_score(embedding, guidance_model)
        return score


# iterates through rule and scores them
# returns list of tuples, each containing rule, score, and index
def score_goal(goal, rules_list, index, model, guidance_model):
    result = []
    for rule in rules_list:
        result.append(
            (rule, score_rule_query(goal, rule, model, guidance_model), index)
        )
    return result


def backwardchainguided(
    KB: knowledgebase.KnowledgeBase,
    path_obj: knowledgebase.Path,
    max_depth: reasoner.MaxDepth,
    model: nnunifier.NeuralNet | None,
    guidance_model: nnreasoner.NeuralNet,
    depth_list: list,
    reasoner_name: reasoner_name_type,
    trace_file,
    alt_select=True,
    use_min_score=True,
    goal_pruning=True,
):
    """Execute a query using the guided reasoner. Uses backward chaining code. Checks depth,
    evaluates body of G, scores each Atom, sorts scores in descending order, checks each rule,
    tries to unify rules with queries

    :param KB: The knowledge base
    :param path_obj: The query
    :param max_depth: Depth limit of search
    :param model: Embedding model
    :param guidance_model: Reasoning model
    :param depth_list:
    :param reasoner_name: Name of reasoner
    :param alt_select: Boolean to determine if alternate selection method should be used
    :param num_pred:
    :param num_var:
    :param num_const:
    :param do_trace: Boolean to determine if trace info should be output.
    :return: False or a dictionary of variable bindings.
    """

    query = path_obj.node
    vars = set()
    for arg in query.arguments:
        if isinstance(arg, Variable):
            vars.add(copy(arg))
    vars = list(vars)
    G = knowledgebase.Rule(
        Atom(Predicate(len(vars), "yes"), copy(vars)), [query])

    t = process_time()

    # Trying out use_min_score. Must manually change this value currently.

    if alt_select:
        success, bindings = backwardmainguidedalt(
            KB,
            G,
            set(vars),
            path_obj,
            max_depth,
            model,
            guidance_model,
            depth_list,
            reasoner_name,
            trace_file,
            use_min_score,
            goal_pruning,
            t
        )
    else:
        success, bindings = backwardmainguided(
            KB,
            G,
            set(vars),
            path_obj,
            max_depth,
            model,
            guidance_model,
            depth_list,
            reasoner_name,
            trace_file,
            use_min_score,
            goal_pruning,
            t
        )
    clear_line()
    if not success:
        print("Query failed!!!")
        open(trace_file, 'a').write("Query failed!!!\n")
    return success, bindings


def backwardmainguided(
    KB: knowledgebase.KnowledgeBase,
    G: knowledgebase.Rule,
    vars: set[Variable],
    path_obj: knowledgebase.Path,
    max_depth: reasoner.MaxDepth,
    model: nnunifier.NeuralNet | None,
    guidance_model: nnreasoner.NeuralNet,
    depth_list: list,
    reasoner_name: reasoner_name_type,
    trace_file,
    use_min_score=True,
    goal_pruning=True,
    start_time=0.0,
):
    """Execute one step of a query using the guided reasoner. By default"""
    max_depth.num_nodes += 1
    do_trace = True

    # if max_depth.num_nodes % 1000 == 1:
    #     diff = process_time() - start_time + 0.0001
    #     clear_line()
    #     print("\r", int(max_depth.num_nodes / diff), '\t',
    #           max_depth.num_nodes, end="", flush=True)

    if max_depth.num_nodes % 5000 == 1 or (KB.length > 250 and max_depth.num_nodes % 1000 == 1):
        diff = process_time() - start_time
        print_progress_bar(max_depth.num_nodes, NODE_MAX,
                           shown='percent', suffix=f'to max depth ({int(max_depth.num_nodes / diff) if diff > 0 else "-"} nps)', length=25)

    if (
        path_obj.depth > int(
            max_depth.max * 1.5) or max_depth.num_nodes > NODE_MAX
    ):
        return False, {}

    if G.body:
        valid_rules: list[tuple[knowledgebase.Rule,
                                float, int, dict | bool]] = []
        goal_selection: dict[int, list[tuple[knowledgebase.Rule,
                                             float, int, dict | bool]]] = {}
        score_list: list = [None] * len(G.body)

        # Goes through each rule in our Knowledge Base, determines which ones unify to our rule,
        # then iterates through each valid rule that unifies. Queries are scored with respect
        # to the reasoner method chosen, then concatenated to a list.
        for i, arg in enumerate(G.body):
            for rule in KB.rule_by_pred[arg.predicate]:
                rule_1: knowledgebase.Rule = copy(rule)
                reasoner.standardize(rule_1, path_obj.depth)
                subst = cache.unify_memoized(rule_1.head, arg)
                # subst = reasoner.old_unify(rule_1.head, arg)

                if not isinstance(subst, dict):
                    continue

                score = (
                    score_rule_query(arg, rule, model, guidance_model)
                    if reasoner_name == "unity" or reasoner_name == "autoencoder"
                    else score_rule_query_termwalk(
                        arg, rule, guidance_model
                    )
                    if reasoner_name == "termwalk"
                    else score_rule_query_chainbased(
                        arg, rule, guidance_model
                    )
                    if reasoner_name == "chainbased"
                    else None
                )

                if not score or (use_min_score and score < MIN_RULE_SCORE):
                    continue

                addition = (rule_1, score, i, subst)
                valid_rules.append(addition)

                if goal_pruning:
                    if i in goal_selection:
                        goal_selection[i].append(addition)
                    else:
                        goal_selection[i] = [addition]
                    score_list[i] = max(
                        score, score_list[i]) if score_list[i] else score

        if goal_pruning and goal_selection:
            valid_rules = goal_selection[score_list.index(
                min(score_list, key=lambda x: x if x else float("inf")))]

        valid_rules.sort(key=lambda x: x[1], reverse=True)

        with open(trace_file, 'a') as f:
            for r in valid_rules:
                best_rule, best_score, a, best_subst = r
                # f.write("Best substitution: {}\n".format(best_subst))
                # f.write("Best rule: {}\n".format(best_rule))
                # trace_old_body = sorted(G.body, key=lambda x: str(x))
                # trace_new_body = sorted(
                #     new_G.body, key=lambda x: str(x))
                # f.write(
                #     "Goal step: {} --> {}\n".format(trace_old_body, trace_new_body))
                # if valid_rules.index(best_goal) > 0:
                #     f.write("({}) Redo: {} (one of {} subgoals)\n".format(
                #         path_obj.depth, a, len(G.body)))
                # else:
                #     f.write("({}) Call: {} (one of {} subgoals)\n".format(
                #         path_obj.depth, a, len(G.body)))
                f.write(str(best_rule.head)+" :- " + str(best_rule.body) + ", " +
                        str(best_score) + ", " + str(G.body[a]) + ", " + str(best_subst))
                f.write("\n")
            f.write("\n\n\n")

        if valid_rules:  # added this if statement in case valid_rules was left empty
            for best_goal in valid_rules:
                body = copy(
                    G.body
                )  # copy the subgoals and pop matching goal, this will be used with the rule's body later
                best_rule, best_score, a, best_subst = best_goal
                # print('bod:', body)
                a1 = body[a]
                body = list(set(body) - set([a1]))

                if use_min_score and best_score < MIN_RULE_SCORE:
                    break  # exit the loop, as all remaining rules will have even lower scores

                # If the current atom in our program unifies with our given rule, that means the query should proceed.
                # Unification is determined through previous iterations.
                if isinstance(
                    best_subst, bool
                ):  # had to makes sure subst is not boolean to avoid AttributeError: 'bool' object has no attribute 'keys' error;
                    continue

                new_body = best_rule.body + body
                new_leaf = path_obj.get_leaf(best_rule, None)
                new_G = knowledgebase.Rule(
                    reasoner.dosubst(G.head, best_subst),
                    [reasoner.dosubst(atom, best_subst)
                     for atom in new_body],
                )

                # if do_trace:
                #     with open(trace_file, 'a') as f:
                #         # f.write("Best substitution: {}\n".format(best_subst))
                #         # f.write("Best rule: {}\n".format(best_rule))
                #         # trace_old_body = sorted(G.body, key=lambda x: str(x))
                #         # trace_new_body = sorted(
                #         #     new_G.body, key=lambda x: str(x))
                #         # f.write(
                #         #     "Goal step: {} --> {}\n".format(trace_old_body, trace_new_body))
                #         # if valid_rules.index(best_goal) > 0:
                #         #     f.write("({}) Redo: {} (one of {} subgoals)\n".format(
                #         #         path_obj.depth, a1, len(G.body)))
                #         # else:
                #         #     f.write("({}) Call: {} (one of {} subgoals)\n".format(
                #         #         path_obj.depth, a1, len(G.body)))
                #         f.write(str(best_goal[1])+", " + str(best_goal[3]))
                #         f.write("\n")
                # elif do_trace and max_depth.num_nodes == TRACE_MAX:
                #     with open(trace_file, 'a') as f:
                #         f.write("...\n")

                success, bindings = backwardmainguided(
                    KB,
                    new_G,
                    vars,
                    new_leaf,
                    max_depth,
                    model,
                    guidance_model,
                    depth_list,
                    reasoner_name,
                    trace_file,
                    use_min_score,
                    goal_pruning,
                    start_time,
                )

                if success:
                    return success, bindings
        # if the code gets here, none of the valid rules worked
        return False, {}
    else:  # G.body is empty, should only get here when successful
        if path_obj.depth < max_depth.max:
            max_depth.set(path_obj.depth)
        depth_list.append(path_obj.depth)
        global DEBUG
        if DEBUG:
            print("Solution: ")
            path_obj.print_rule_path()
            print()
        return True, {list(vars)[i]: G.head.arguments[i] for i in range(len(vars))}


def backwardmainguidedalt(
    KB: knowledgebase.KnowledgeBase,
    G: knowledgebase.Rule,
    vars: set[Variable],
    path_obj: knowledgebase.Path,
    max_depth: reasoner.MaxDepth,
    model: nnunifier.NeuralNet | None,
    guidance_model: nnreasoner.NeuralNet,
    depth_list: list,
    reasoner_name: reasoner_name_type,
    trace_file,
    use_min_score=True,
    goal_pruning=True,
    start_time=0.0,
):
    """Execute one step of a query using the guided reasoner. By default"""
    max_depth.num_nodes += 1
    global DEBUG
    do_trace = DEBUG

    # if max_depth.num_nodes % 1000 == 1:
    #     diff = process_time() - start_time + 0.0001
    #     clear_line()
    #     print("\r", int(max_depth.num_nodes / diff), '\t',
    #           max_depth.num_nodes, end="", flush=True)

    if max_depth.num_nodes % 5000 == 1 or (KB.length > 250 and max_depth.num_nodes % 1000 == 1):
        diff = process_time() - start_time
        print_progress_bar(max_depth.num_nodes, NODE_MAX,
                           shown='percent', suffix=f'to max nodes ({int(max_depth.num_nodes / diff) if diff > 0 else "-"} nps)', length=25)

    if (
        path_obj.depth > int(
            max_depth.max * 1.5) or max_depth.num_nodes > NODE_MAX
    ):
        return False, {}

    if G.body:
        valid_rules: list[rule_type] = []
        goal_selection: dict[int, list[rule_type]] = {}
        score_list: list[float] = [1] * len(G.body)

        # Goes through each rule in our Knowledge Base, determines which ones unify to our rule,
        # then iterates through each valid rule that unifies. Queries are scored with respect
        # to the reasoner method chosen, then concatenated to a list.
        for i, arg in enumerate(G.body):
            goal_rules, max_score = grc.generate_goals(
                arg, KB, path_obj.depth)
            valid_rules += goal_rules
            goal_selection[i] = goal_rules
            score_list[i] = max_score

        if goal_pruning and goal_selection:
            valid_rules = goal_selection[score_list.index(
                min(score_list))]
        # with open(trace_file, 'a') as f:
        #     for r in valid_rules:
        #         best_rule, best_score, a, best_subst = r
        #         # f.write("Best substitution: {}\n".format(best_subst))
        #         # f.write("Best rule: {}\n".format(best_rule))
        #         # trace_old_body = sorted(G.body, key=lambda x: str(x))
        #         # trace_new_body = sorted(
        #         #     new_G.body, key=lambda x: str(x))
        #         # f.write(
        #         #     "Goal step: {} --> {}\n".format(trace_old_body, trace_new_body))
        #         # if valid_rules.index(best_goal) > 0:
        #         #     f.write("({}) Redo: {} (one of {} subgoals)\n".format(
        #         #         path_obj.depth, a, len(G.body)))
        #         # else:
        #         #     f.write("({}) Call: {} (one of {} subgoals)\n".format(
        #         #         path_obj.depth, a, len(G.body)))
        #         f.write(str(best_rule.head)+" :- " + str(best_rule.body) + ", " +
        #                 str(best_score) + ", " + str(a) + ", " + str(best_subst))
        #         f.write("\n")
        #     f.write("\n\n\n")
        if valid_rules:  # added this if statement in case valid_rules was left empty
            for best_goal in valid_rules:
                body = copy(
                    G.body
                )  # copy the subgoals and pop matching goal, this will be used with the rule's body later
                best_rule, best_score, a, best_subst = best_goal
                body = list(set(body) - set([a]))
                # body.remove(a)

                if use_min_score and best_score < MIN_RULE_SCORE:
                    break  # exit the loop, as all remaining rules will have even lower scores

                # If the current atom in our program unifies with our given rule, that means the query should proceed.
                # Unification is determined through previous iterations.
                if isinstance(
                    best_subst, bool
                ):  # had to makes sure subst is not boolean to avoid AttributeError: 'bool' object has no attribute 'keys' error;
                    continue

                new_body = best_rule.body + body
                new_leaf = path_obj.get_leaf(best_rule, None)
                new_G = knowledgebase.Rule(
                    reasoner.dosubst(G.head, best_subst),
                    [reasoner.dosubst(atom, best_subst)
                     for atom in new_body],
                )

                if do_trace and max_depth.num_nodes < TRACE_MAX:
                    print("Best substitution:", best_subst)
                    print('Best rule:', best_rule)
                    print('Goal step:', G.body, '-->', new_G.body)
                    if valid_rules.index(best_goal) > 0:
                        print("(" + str(path_obj.depth) + ") Redo: " + str(a) + " (one of " + str(
                            len(G.body)) + " subgoals)")
                    else:
                        print("(" + str(path_obj.depth) + ") Call: " + str(a) + " (one of " + str(
                            len(G.body)) + " subgoals)")
                    print()
                elif do_trace and max_depth.num_nodes == TRACE_MAX:
                    print('...')

                # if do_trace:
                #     with open(trace_file, 'a') as f:
                #         # f.write("Best substitution: {}\n".format(best_subst))
                #         # f.write("Best rule: {}\n".format(best_rule))
                #         # trace_old_body = sorted(G.body, key=lambda x: str(x))
                #         # trace_new_body = sorted(
                #         #     new_G.body, key=lambda x: str(x))
                #         # f.write(
                #         #     "Goal step: {} --> {}\n".format(trace_old_body, trace_new_body))
                #         # if valid_rules.index(best_goal) > 0:
                #         #     f.write("({}) Redo: {} (one of {} subgoals)\n".format(
                #         #         path_obj.depth, a, len(G.body)))
                #         # else:
                #         #     f.write("({}) Call: {} (one of {} subgoals)\n".format(
                #         #         path_obj.depth, a, len(G.body)))
                #         f.write(str(best_rule.head)+" :- " + str(best_rule.body) + ", " +
                #                 str(best_score) + ", " + str(a) + ", " + str(best_subst))
                #         f.write("\n")
                # elif do_trace and max_depth.num_nodes == TRACE_MAX:
                #     with open(trace_file, 'a') as f:
                #         f.write("...\n")

                success, bindings = backwardmainguidedalt(
                    KB,
                    new_G,
                    vars,
                    new_leaf,
                    max_depth,
                    model,
                    guidance_model,
                    depth_list,
                    reasoner_name,
                    trace_file,
                    use_min_score,
                    goal_pruning,
                    start_time,
                )

                if success:
                    return success, bindings
        # if the code gets here, none of the valid rules worked
        return False, {}
    else:  # G.body is empty, should only get here when successful
        if path_obj.depth < max_depth.max:
            max_depth.set(path_obj.depth)
        depth_list.append(path_obj.depth)
        if DEBUG:
            print("Solution: ")
            path_obj.print_rule_path()
            print()
        return True, {list(vars)[i]: G.head.arguments[i] for i in range(len(vars))}

# backwardchainbasic methods similar to previous methods without using scores


def backwardchainbasic(
    KB: knowledgebase.KnowledgeBase,
    path_obj: knowledgebase.Path,
    max_depth,
    depth_list,
    do_trace=False,
):
    """Execute a query using a standard backward-chaining reasoner.

    :param KB:
    :param path_obj:
    :param max_depth:
    :param depth_list:
    :param do_trace:
    :return:
    """

    # global globalCount
    # global exitGlobalCount
    # global keepGoing
    # global queryList
    query = path_obj.node
    vars = set()
    for arg in query.arguments:
        if isinstance(arg, Variable):
            vars.add(copy(arg))
    vars = list(vars)
    #    if G not in queryList:           # where is G defined
    #        queryList.append(G)
    G = knowledgebase.Rule(
        Atom(Predicate(len(vars), "yes"), copy(vars)), [query])
    # if globalCount < 20 and keepGoing == True:
    #     if callFlag == True:
    #         print(str(globalCount) + " Call: " + str(queryList[-1]))
    #         globalCount += 1
    #         exitGlobalCount += 1
    # else:
    #     exitGlobalCount += 1
    # if globalCount >= 20:
    #     globalCount = 0
    #     keepGoing = False
    success, bindings = backwardmainbasic(
        KB, G, vars, path_obj, max_depth, depth_list, do_trace
    )
    clear_line()
    if not success:
        print("Query failed!!!")
    return bindings


def backwardmainbasic(
    KB: knowledgebase.KnowledgeBase,
    G: knowledgebase.Rule,
    vars,
    path_obj: knowledgebase.Path,
    max_depth: reasoner.MaxDepth,
    depth_list,
    do_trace=False,
):
    """Execute a step of the standard backward-chaining reasoner."""

    max_depth.num_nodes += 1
    if (
        path_obj.depth > int(
            max_depth.max * 1.5) or max_depth.num_nodes > NODE_MAX
    ):  # depth limiter
        if do_trace and max_depth.num_nodes < TRACE_MAX:
            print("(" + str(path_obj.depth) + ") Fail: Exceeded depth limit!!!")
        return False, {}

    # signal that work is proceeding
    if max_depth.num_nodes % 100000 == 1:
        # print(".", end="", flush=True)
        print_progress_bar(
            max_depth.num_nodes, NODE_MAX, shown='percent', suffix='to max depth', length=25)

    if G.body:
        a1 = G.body.pop(0)

        if do_trace and (
            max_depth.num_nodes <= TRACE_MAX or path_obj.depth <= TRACE_UP_TO_MIN
        ):
            print(
                "("
                + str(path_obj.depth)
                + ") Call: "
                + str(a1)
                + " [first of "
                + str(len(G.body) + 1)
                + " subgoals]"
            )
        path_obj.set_node(a1)
        no_ans = True
        first_rule = True

        for rule in KB.rule_by_pred[a1.predicate]:
            # standardizes rule
            rule_1 = copy(rule)

            reasoner.standardize(rule_1, path_obj.depth)
            # unifies atom and rule head
            subst = cache.unify_memoized(a1, rule_1.head)
            # if unification fails, continue on to the next rule

            # Generates new rule to pass into next iteration of backwardmainbasic for backward chaining reasoner.
            if not isinstance(subst, dict) or not subst:
                continue

            new_body = rule_1.body + G.body
            new_leaf = path_obj.get_leaf(rule, None)
            new_G = knowledgebase.Rule(
                reasoner.dosubst(G.head, subst),
                [reasoner.dosubst(atom, subst) for atom in new_body],
            )
            new_G_Head = new_G.head
            G_Head = G.head
            if do_trace and (
                max_depth.num_nodes < TRACE_MAX or path_obj.depth <= TRACE_UP_TO_MIN
            ):
                if not first_rule:
                    print("(" + str(path_obj.depth) + ") Redo: " + str(a1))
                print("\tRule: " + str(rule_1))

            success, bindings = backwardmainbasic(
                KB, new_G, vars, new_leaf, max_depth, depth_list, do_trace
            )
            #      yield ret_val
            first_rule = False
            if success:
                # if ret_val != False:
                #    print(str(globalCount) + " Exit: " + str(finalRule))
                no_ans = False
                if do_trace and max_depth.num_nodes < TRACE_MAX:
                    print("(" + str(path_obj.depth) +
                          ") Exit: " + str(bindings))

                return success, bindings

        if no_ans:
            if do_trace and max_depth.num_nodes < TRACE_MAX:
                print("(" + str(path_obj.depth) + ") Fail")
            return False, {}
    else:  # G.body is empty, should only get here when successful
        if path_obj.depth < max_depth.max:
            max_depth.set(path_obj.depth)
        # yield {vars[i] : G.head.arguments[i] for i in range(len(vars))}
        # do we need the depth_list parameter?
        depth_list.append(path_obj.depth)  # added
        return True, {vars[i]: G.head.arguments[i] for i in range(len(vars))}


# uses backward chaining guided reasoning to compute # of nodes visited to reach each query
# returns average number of nodes visited across all queries
def guided(
    queries: list[Atom],
    model: nnunifier.NeuralNet | None,
    guidance_model: nnreasoner.NeuralNet,
    reasoner_name: reasoner_name_type,
    data: list,
    kb_file="randomKB.txt",
    use_min_score=True,
    use_goal_pruning=True,
):
    global globalCount
    global exitGlobalCount
    global timeToExecute

    fail_queries = 0

    if reasoner_name == "unity":
        f = open("unification_nodes_traversed.csv", "w", newline="")
    elif reasoner_name == "autoencoder":
        f = open("autoencoder_nodes_traversed.csv", "w", newline="")
    elif reasoner_name == "chainbased":
        f = open("chainbased_nodes_traversed.csv", "w", newline="")
    elif reasoner_name == "termwalk":
        f = open("termwalk_nodes_traversed.csv", "w", newline="")
    elif reasoner_name == "standard":
        f = open("standard_nodes_traversed.csv", "w", newline="")
    else:
        raise ValueError
    writer = csv.writer(f)
    headerColumns = ["Query", "Nodes Traversed"]
    writer.writerow(headerColumns)
    KB = parse_KB_file(kb_file)
    data_dict = []
    guided_count_total = []
    use_alt = True
    log_file = 'trace_log.txt' if use_alt else 'trace_log-old.txt'
    # i = 0
    for i in range(len(queries)):
        # for i in range(10):
        # for i in [14]:
        time_start = process_time()
        query = queries[i]
        print("Query " + str(i+1) + ": " + str(query))
        with open(log_file, 'a') as f:
            f.write("Query " + str(i+1) + ": " + str(query) + "\n")
        i += 1
        path_guide = knowledgebase.Path(query, None, None, 0)
        max_depth_guide = reasoner.MaxDepth(10)
        depths = []
        # Every iteration of our backward chaining reasoner begins with executing the backwardchainguided method.
        success, guided_answers = backwardchainguided(
            KB,
            path_guide,
            max_depth_guide,
            model,
            guidance_model,
            depths,
            reasoner_name,
            log_file,
            use_alt,
            use_min_score,
            use_goal_pruning,
        )
        if not success:
            fail_queries = fail_queries + 1

        guided_count_total.append(max_depth_guide.num_nodes)

        time_end = process_time()
        timeToExecute += time_end - time_start
        exitGlobalCount = 0
        globalCount = 0

        nodesTraversed = max_depth_guide.num_nodes
        row = [str(query), str(nodesTraversed)]
        writer.writerow(row)

        if depths != []:
            min_dep = min(depths)
        else:
            min_dep = 0

        nodes = max_depth_guide.num_nodes
        t = time.strftime("%H:%M:%S",
                          time.gmtime(time_end - time_start))
        print(
            f"{max_depth_guide.num_nodes} :: {min_dep} - {t} ({int(max_depth_guide.num_nodes/(time_end - time_start)) if (time_end - time_start) > 0 else '-'} nps)\n")
        open(log_file, 'a').write(
            f"{max_depth_guide.num_nodes} :: {min_dep} - {t} ({int(max_depth_guide.num_nodes/(time_end - time_start)) if (time_end - time_start) > 0 else '-'} nps)\n\n")

        if reasoner_name == "unity":
            data_dict.append(
                {
                    "query": i,
                    "unity reasoner": reasoner_name,
                    "unity nodes explored": nodes,
                    "unity min depth": min_dep,
                    "success": success,
                    "time": time_end - time_start,
                }
            )
        elif reasoner_name == "autoencoder":
            data_dict.append(
                {
                    "query": i,
                    "auto reasoner": reasoner_name,
                    "auto nodes explored": nodes,
                    "auto min depth": min_dep,
                    "success": success,
                }
            )
        elif reasoner_name == "chainbased":
            data_dict.append(
                {
                    "query": i,
                    "chainbased reasoner": reasoner_name,
                    "chainbased nodes explored": nodes,
                    "chainbased min depth": min_dep,
                    "success": success,
                }
            )
        elif reasoner_name == "termwalk":
            data_dict.append(
                {
                    "query": i,
                    "termwalk reasoner": reasoner_name,
                    "termwalk nodes explored": nodes,
                    "termwalk min depth": min_dep,
                    "success": success,
                }
            )
        else:
            raise ValueError
        # print()

    f.close()
    guide_mean_total = sum(guided_count_total) / len(guided_count_total)
    print(f"guided: {guide_mean_total}")
    if fail_queries > 0:
        print(str(fail_queries) + " queries failed")

    data += data_dict
    return guide_mean_total


# uses backward chaining basic reasoning to compute number of nodes visited to reach each query
# returns average number of nodes visited across all queries
def base(queries, reasoner_name, data, kb_file="randomKB.txt", do_trace=False):
    KB = parse_KB_file(kb_file)
    # i = 0
    base_data_dict = []
    base_count_total = []
    for i in range(len(queries)):
        # for i in range(min(10, len(queries))):
        # for i in [14]:
        query = queries[i]
        print("Query " + str(i+1) + ": " + str(query))
        i += 1
        path_base = knowledgebase.Path(query, None, None, 0)
        max_depth_base = reasoner.MaxDepth(7)
        depths = []
        base_answers = backwardchainbasic(
            KB, path_base, max_depth_base, depths, do_trace
        )
        nodes = max_depth_base.num_nodes

        base_count_total.append(nodes)

        if depths != []:
            min_dep = min(depths)
            print(f"{nodes} :: {min_dep}")
        else:
            min_dep = 0

        print()
        base_data_dict.append(
            {
                "query": i,
                "base reasoner": reasoner_name,
                "base nodes explored": nodes,
                "base min depth": min_dep,
            }
        )

    base_mean = sum(base_count_total) / len(base_count_total)

    data += base_data_dict
    return base_mean


#  loads pre-trained neural network model, generates test queries,
#  and calls base() and guided() to compare performance of two backward chaining algorithms
if __name__ == "__main__":
    aparser = argparse.ArgumentParser(description="Run the experiment.")
    aparser.add_argument(
        "--alt_select",
        action="store_true",
        help="Use alternative control strategy of getting best goal from worst rule.",
    )
    aparser.add_argument("-s", "--standard", action="store_true")
    aparser.add_argument("-u", "--unifier", action="store_true")
    aparser.add_argument(
        "--kb", default="randomKB.txt", help="Name of file containing knowledge base"
    )
    aparser.add_argument(
        "--qfile", default="test_queries.txt", help="Name of file containing queries"
    )
    aparser.add_argument(
        "--unifier_model_path",
        default="rKB_model.pth",
        help="The path to the unification embeddings model. By default, rKB_model.pth.",
    )
    # aparser.add_argument("--unifier_guidance_model_path", default="rule_classifier.pth", help="The path to the guided reasoner model trained using unification embeddings. By default, rule_classifier.pth.")
    aparser.add_argument(
        "--unifier_guidance_model_path",
        default="uni_mr_model.pt",
        help="The path to the guided reasoner model trained using unification embeddings. By default, rule_classifier.pth.",
    )
    aparser.add_argument("-a", "--autoencoder", action="store_true")
    aparser.add_argument(
        "--auto_model_path",
        default="auto_encoder.pth",
        help="The path to the autoencoder embeddings model. By default, auto_encoder.pth.",
    )
    aparser.add_argument(
        "--auto_guidance_model_path",
        default="auto_rule_classifier.pth",
        help="The path to the guided reasoner model trained using autoencoder embeddings. By default, auto_rule_classifier.pth.",
    )
    aparser.add_argument("-t", "--termwalk", action="store_true")
    # aparser.add_argument("--termwalk_guidance_model_path", default="termwalk_rule_classifier.pth", help="The path to the guided reasoner model trained using termwalk embeddings. By default, termwalk_rule_classifier.pth.")
    aparser.add_argument(
        "--termwalk_guidance_model_path",
        default="tw_mr_model.pt",
        help="The path to the guided reasoner model trained using termwalk embeddings. By default, termwalk_rule_classifier.pth.",
    )
    aparser.add_argument("--predicates", default=10, type=int)
    aparser.add_argument("--variables", default=10, type=int)
    aparser.add_argument("--constants", default=100, type=int)
    aparser.add_argument("-c", "--chainbased", action="store_true")
    # aparser.add_argument("--chainbased_guidance_model_path", default="chainbased_rule_classifier.pth", help="The path to the guided reasoner model trained using chainbased embeddings. By default, chainbased_rule_classifier.pth.")
    aparser.add_argument(
        "--chainbased_guidance_model_path",
        default="cb_mr_model.pt",
        help="The path to the guided reasoner model trained using chainbased embeddings. By default, chainbased_rule_classifier.pth.",
    )
    aparser.add_argument(
        "--trace", action="store_true", help="Output a trace of each query"
    )
    aparser.add_argument(
        "--no_min_score",
        action="store_true",
        help="Indicates that all matching rules will be attempted",
    )
    aparser.add_argument(
        "--no_goal_pruning",
        action="store_true",
        help="Indicates that goal pruning will not be used",
    )
    aparser.add_argument(
        "--load_vocab",
        action="store_true",
        help="Load vocab from file instead of generating it from the KB",
    )
    aparser.add_argument(
        "--vocab_file",
        default="vocab",
        help="Path to load initial vocabulary from. If not specified, a vocabulary will be generated from the KB.",
    )
    aparser.add_argument("-e", "--embed_size", type=int, default=50,
                         help="Embed size. Defaults to 50")

    args = aparser.parse_args()
    embed_size = args.embed_size  # get_embed_size(vocab)

    if args.unifier:
        if not (
            os.path.isfile(args.unifier_model_path)
            and os.path.isfile(args.unifier_guidance_model_path)
        ):
            print("No file found at path for unifier model")
            sys.exit(1)
    if args.autoencoder:
        if not (
            os.path.isfile(args.auto_model_path)
            and os.path.isfile(args.auto_guidance_model_path)
        ):
            print("No file found at path for autoencoder model")
            sys.exit(1)
    if args.termwalk:
        if not os.path.isfile(args.termwalk_guidance_model_path):
            print("No file found at path for termwalk model")
            sys.exit(1)
    if args.chainbased:
        if not os.path.isfile(args.chainbased_guidance_model_path):
            print("No file found at path for chain-based model")
            sys.exit(1)
    if args.load_vocab:
        vocab.init_from_vocab(args.vocab_file)
    else:
        vocab.init_from_kb(parse_KB_file(args.kb))

    input_size = len(vocab.predicates) + (
        (len(vocab.variables) + len(vocab.constants)) * vocab.maxArity
    )

    # NOTE: choosing a KB other than randomKB.txt for the guided reasoner can have unpredicatable results
    # In particular, there is no code to handle embedding of vocabularies different from the random vocab

    if args.kb:
        if not os.path.isfile(args.kb):
            print("Invalid KB file " + os.path.abspath(args.kb))
            sys.exit(1)
    if args.qfile:
        if not os.path.isfile(args.qfile):
            print("Invalid query file " + os.path.abspath(args.qfile))
            sys.exit(1)

    test_queries = kbparser.parse_KB_file(args.qfile).rules
    queries = [query.head for query in test_queries]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device} device")
    data1, data2, data3, data4, data5 = [], [], [], [], []

    # run the selected models on the test queries: standard, unifier, autoencoder, termwalk, and/or chain-based

    use_min_score = not args.no_min_score
    if not use_min_score:
        print("Not using a minimum score for rules")

    global DEBUG
    DEBUG = args.trace

    # separate booleans for basic reasoner
    if args.standard:
        print("STANDARD\n")
        base(queries, "standard", data1, args.kb, args.trace)

        with open("standard_data.csv", mode="w", newline="") as file:
            fieldnames = [
                "query",
                "base reasoner",
                "base nodes explored",
                "base min depth",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            if file.tell() == 0:
                writer.writeheader()
            for row in data1:
                writer.writerow(row)

    # assumes that the appropriate embeddings and meta reasoning models have been trained already
    if args.unifier:
        print("UNITY")
        if not args.no_goal_pruning:
            print("Goal Pruning Enabled")
        print("\tEmbedding Model: " + args.unifier_model_path)
        print("\tReasoning Model: " + args.unifier_guidance_model_path)
        model_path = args.unifier_model_path
        # base(queries, "unity")
        model = nnunifier.NeuralNet(
            input_size,
            nnunifier.hidden_size1,
            nnunifier.hidden_size2,
            embed_size,
        ).to(device)
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device(device)))
        guidance_model = nnreasoner.NeuralNet(
            nnreasoner.hidden_size1, nnreasoner.hidden_size2, nnreasoner.num_classes
        ).to(device)
        guidance_model.load_state_dict(
            torch.load(
                args.unifier_guidance_model_path, map_location=torch.device(
                    device)
            )
        )
        guidedm = guided(
            queries,
            model,
            guidance_model,
            "unity",
            data2,
            args.kb,
            use_min_score,
            not args.no_goal_pruning,
        )
        print("Time took to execute program: " + str(timeToExecute))
        fn = ("unity_data-i" if not args.no_goal_pruning else "unity_data") + \
            f"-{len(vocab.predicates)}-{len(vocab.constants)}-{vocab.maxArity}-{embed_size}.csv"
        with open(fn, mode="w", newline="") as file:
            fieldnames = [
                "query",
                "unity reasoner",
                "unity nodes explored",
                "unity min depth",
                "success",
                "time",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            if file.tell() == 0:
                writer.writeheader()
            for row in data2:
                writer.writerow(row)

    if args.autoencoder:
        model_path = args.auto_model_path
        print("AUTO")
        print("\tEmbedding Model: " + model_path)
        print("\tReasoning Model: " + args.auto_guidance_model_path)
        whole_model = autoencoder.NeuralNet().to(device)
        whole_model.load_state_dict(
            torch.load(model_path, map_location=torch.device(device))
        )
        whole_model.eval()
        model = whole_model.encoder
        guidance_model = nnreasoner.NeuralNet(
            nnreasoner.hidden_size1, nnreasoner.hidden_size2, nnreasoner.num_classes
        ).to(device)
        guidance_model.load_state_dict(
            torch.load(args.auto_guidance_model_path,
                       map_location=torch.device(device))
        )
        guidedm = guided(
            queries,
            model,
            guidance_model,
            "autoencoder",
            data3,
            args.kb,
            args.alt_select,
            args.trace,
            use_min_score,
        )

        with open("auto_data.csv", mode="w", newline="") as file:
            fieldnames = [
                "query",
                "auto reasoner",
                "auto nodes explored",
                "auto min depth",
                "success",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            if file.tell() == 0:
                writer.writeheader()
            for row in data3:
                writer.writerow(row)

    if args.termwalk:
        print("TERMWALK")
        print("\tReasoning Model: " + args.termwalk_guidance_model_path)
        model = None
        guidance_model = nnreasoner.NeuralNet(
            nnreasoner.hidden_size1, nnreasoner.hidden_size2, nnreasoner.num_classes
        ).to(device)
        guidance_model.load_state_dict(
            torch.load(
                args.termwalk_guidance_model_path, map_location=torch.device(
                    device)
            )
        )
        guidedm = guided(
            queries,
            model,
            guidance_model,
            "termwalk",
            data4,
            args.kb,
            args.alt_select,
        )
        with open("termwalk_data.csv", mode="w", newline="") as file:
            fieldnames = [
                "query",
                "termwalk reasoner",
                "termwalk nodes explored",
                "termwalk min depth",
                "success",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            if file.tell() == 0:
                writer.writeheader()
            for row in data4:
                writer.writerow(row)

    if args.chainbased:
        print("CHAINBASED")
        print("\tReasoning Model: " + args.chainbased_guidance_model_path)
        model = None
        guidance_model = nnreasoner.NeuralNet(
            nnreasoner.hidden_size1, nnreasoner.hidden_size2, nnreasoner.num_classes
        ).to(device)
        guidance_model.load_state_dict(
            torch.load(
                args.chainbased_guidance_model_path, map_location=torch.device(
                    device)
            )
        )
        guidedm = guided(
            queries,
            model,
            guidance_model,
            "chainbased",
            data5,
            args.kb,
            args.alt_select,
            args.trace,
            use_min_score,
        )
        with open("chainbased_data.csv", mode="w", newline="") as file:
            fieldnames = [
                "query",
                "chainbased reasoner",
                "chainbased nodes explored",
                "chainbased min depth",
                "success",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            if file.tell() == 0:
                writer.writeheader()
            for row in data5:
                writer.writerow(row)

    if args.standard and args.unifier and args.autoencoder:
        all_data = []
        for i in range(len(data1)):
            combined_dict = {}
            for d in [data1[i], data2[i], data3[i]]:
                combined_dict.update(d)
            all_data.append(combined_dict)

        with open("data.csv", mode="w", newline="") as file:
            fieldnames = [
                "query",
                "base reasoner",
                "base nodes explored",
                "base min depth",
                "unity reasoner",
                "unity nodes explored",
                "unity min depth",
                "auto reasoner",
                "auto nodes explored",
                "auto min depth",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            if file.tell() == 0:
                writer.writeheader()
            for row in all_data:
                writer.writerow(row)

    # check for each random_fact if it is already in KB
    # take a single query and go through 1 line at a time

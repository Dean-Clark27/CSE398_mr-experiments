# **** This class is deprecated *****
# All useful functionality has been moved to reasoner.py and evaluate.py
# For a while, the only functions being used were backwardchainbasic() and backwardmainbasic()

import sys
import chainbased
import termwalk
import argparse
import csv
import autoencoder
import kbencoder
import nnreasoner
from time import process_time
from basictypes import Atom, Predicate, Variable
from copy import deepcopy, copy
import nnunifier
import kbparser
import numpy as np
from kbparser import parse_KB_file
import knowledgebase
import reasoner
import torch
import os

from vocab import Vocabulary

# need this to fix interrupt issues, must be before any scipy is imported
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"


# When this option is used, a rule must have at least the score to be evaluated
MIN_RULE_SCORE = 0.01
# MIN_RULE_SCORE = 0.001
# FALLBACK_DEPTH = 3  # The depth at which the meta-reasoner switches to the standard reasoner
FALLBACK_DEPTH = 5
NODE_MAX = 10_000_000  # on KB500, standard averages as many as 15mil, unity(old) 40mil
DEBUG = False
# The maximum number of reasoning steps that trace information will be output for (per query)
TRACE_MAX = 50
TRACE_UP_TO_MIN = 2  # Level of search shown by trace, even after max nodes is reached

# global timeToExecute
timeToExecute = 0
# checkList = []
# callList = []
# queryList = []

vocab = Vocabulary()

# Alex's comment:
# relevant functions for guided
# score _rule_query - > gets score
# score goal -> list of all possible scores from rules
# guided sorts them all, then goes through them one by one


def score_rule_query(query, rule, model, guidance_model):
    """Evaluates query and rule and returns a score. One-hot encodes query,
    then concatenates embeddings passes embeddings to nnreasoner. Uses get_score
    to return a score

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
        atom = model(torch.FloatTensor(query[0]).to(
            device)).cpu().detach().numpy()
        rule_head = model(torch.FloatTensor(
            query[1]).to(device)).cpu().detach().numpy()
        args = torch.zeros(embed_size).to(device)
        rule_args = query[2]
        for arg in rule_args:
            arg = model(torch.FloatTensor(arg).to(device))
            args = torch.add(args, arg)
        args = args.cpu().detach().numpy()

        embedding = np.concatenate([atom, rule_head, args])

        score = nnreasoner.get_score(embedding, guidance_model)
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
    max_depth,
    model,
    guidance_model,
    depth_list,
    reasoner_name,
    alt_select,
    num_pred,
    num_var,
    num_const,
    do_trace=False,
    use_min_score=True,
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
    # global globalCount
    # global exitGlobalCount
    # global keepGoing
    # global callFlag
    # global queryList
    query = path_obj.node
    vars = set()
    for arg in query.arguments:
        if isinstance(arg, Variable):
            vars.add(copy(arg))
    vars = list(vars)
    G = knowledgebase.Rule(
        Atom(Predicate(len(vars), "yes"), copy(vars)), [query])
    score = None
    # if G not in queryList:
    #    queryList.append(G)
    score = None

    # this code is useless, only results in printing the score of the last matching rule
    # for arg in G.body:
    #     for rule in KB.rule_by_pred[arg.predicate]:
    #         rule_1 = deepcopy(rule)
    #         reasoner.standardize(rule_1,path_obj.depth)
    #         subst = reasoner.unify(arg,rule.head)
    #         if not isinstance(subst,bool):
    #             if reasoner_name == "unity" or reasoner_name == "autoencoder":
    #                 score = score_rule_query(query, rule, model, guidance_model)
    #             elif reasoner_name == "termwalk":
    #                 score = score_rule_query_termwalk(query, rule, guidance_model, num_pred, num_var, num_const)
    #             elif reasoner_name == "chainbased":
    #                 score = score_rule_query_chainbased(query, rule, guidance_model)
    #             else:
    #                 raise ValueError
    # if globalCount < 20 and keepGoing == True:
    #     if callFlag == True:
    #         # print(str(globalCount) + " Call: " + str(queryList[-1]))
    #         # print("Score: " + str(score))
    #         globalCount += 1
    #         exitGlobalCount += 1
    # else:
    #     exitGlobalCount += 1
    # if globalCount >= 20:
    #     globalCount = 0
    #     keepGoing = False
    if not alt_select:
        success, bindings = backwardmainguided(
            KB,
            G,
            vars,
            path_obj,
            max_depth,
            model,
            guidance_model,
            depth_list,
            reasoner_name,
            num_pred,
            num_var,
            num_const,
            do_trace,
            use_min_score,
        )
        if max_depth.num_nodes > 100000:
            print()  # add new line after dots
        if not success:
            print("Query failed!!!")
        return success, bindings
    else:
        success, bindings = backwardmainguidedalt(
            KB,
            G,
            vars,
            path_obj,
            max_depth,
            model,
            guidance_model,
            depth_list,
            reasoner_name,
            num_pred,
            num_var,
            num_const,
            do_trace,
        )
        if max_depth.num_nodes > 100000:
            print()  # add new line after dots
        if not success:
            print("Query failed!!!")
        return success, bindings


def backwardmainguided(
    KB: knowledgebase.KnowledgeBase,
    G: knowledgebase.Rule,
    vars: set[Variable],
    path_obj: knowledgebase.Path,
    max_depth: reasoner.MaxDepth,
    model,
    guidance_model,
    depth_list,
    reasoner_name,
    num_pred,
    num_var,
    num_const,
    do_trace=False,
    use_min_score=True,
):
    """Execute one step of a query using the guided reasoner. By default"""
    # finalRule = None
    max_depth.num_nodes += 1

    # signal that work is proceeding
    if max_depth.num_nodes % 100000 == 0:
        print(".", end="", flush=True)

    # TODO: figure out why we multiply the max depth by 1.5!!!
    #  I think: the 7 is the depth of the hardest query. We wanted 1.5 to give some room to find deeper solutions
    if (
        path_obj.depth > int(
            max_depth.max * 1.5) or max_depth.num_nodes > NODE_MAX
    ):  # depth limiter
        return False, {}

    if G.body:
        # a1 = G.body.pop(0)
        # path_obj.set_node(a1)
        no_ans = True
        valid_rules = []
        i = 0  # will be the index of the goal
        # subStringDict = []
        # Goes through each rule in our Knowledge Base, determines which ones unify to our rule,
        # then iterates through each valid rule that unifies. Queries are scored with respect
        # to the reasoner method chosen, then concatenated to a list.
        for arg in G.body:
            for rule in KB.rule_by_pred[
                arg.predicate
            ]:  # loop to go through rules and check if it unifies w arg before adding to valid_rules
                # standardizes rule
                rule_1 = copy(
                    rule
                )  # only needed if standardize() actually changes its argument
                reasoner.standardize(rule_1, path_obj.depth)
                subst = reasoner.unify(
                    rule_1.head, arg
                )  # should keep vars from arg as long as possible
                if isinstance(subst, dict):  # if it does, add to valid_rules
                    # goalScore = score_goal(G,valid_rules,goalNumber,model,guidance_model)
                    # if goalScore:
                    #    print("Score of Goal: " + str(goalScore))
                    #    goalNumber += 1
                    if reasoner_name == "unity" or reasoner_name == "autoencoder":
                        # changed rule_1 to rule, to make sure that subst will work correctly, added subst
                        valid_rules.append(
                            (
                                rule_1,
                                score_rule_query(
                                    arg, rule, model, guidance_model),
                                i,
                                subst,
                            )
                        )
                    elif reasoner_name == "termwalk":
                        valid_rules.append(
                            (
                                rule_1,
                                score_rule_query_termwalk(
                                    arg,
                                    rule,
                                    guidance_model,
                                    num_pred,
                                    num_var,
                                    num_const,
                                ),
                                i,
                                subst,
                            )
                        )
                    elif reasoner_name == "chainbased":
                        valid_rules.append(
                            (
                                rule_1,
                                score_rule_query_chainbased(
                                    arg, rule, guidance_model),
                                i,
                                subst,
                            )
                        )
                    else:
                        raise ValueError
                else:
                    continue
            i += 1
        valid_rules.sort(key=lambda x: x[1], reverse=True)
        failFlag = False
        # print(G.body)
        if valid_rules:  # added this if statement in case valid_rules was left empty
            rule_seq = 1
            for best_goal in valid_rules:
                rule_1 = best_goal[0]
                score = best_goal[1]
                body = deepcopy(
                    G.body
                )  # copy the subgoals and pop matching goal, this will be used with the rule's body later
                a1 = body.pop(best_goal[2])  # goal that is selected to resolve
                subst = best_goal[
                    3
                ]  # we already found the substitution and save it when scoring

                if use_min_score and score < MIN_RULE_SCORE:
                    if do_trace and max_depth.num_nodes < TRACE_MAX:
                        print(
                            "("
                            + str(path_obj.depth)
                            + ") Call: "
                            + str(a1)
                            + " (one of "
                            + str(len(G.body))
                            + " subgoals)"
                        )
                        print("(" + str(path_obj.depth) +
                              ") Fail [score < min]")
                    break  # exit the loop, as all remaining rules will have even lower scores

                # If the current atom in our program unifies with our given rule, that means the query should proceed.
                # Unification is determined through previous iterations.
                if not isinstance(
                    subst, bool
                ):  # had to makes sure subst is not boolean to avoid AttributeError: 'bool' object has no attribute 'keys' error;
                    new_body = rule_1.body + body
                    new_leaf = path_obj.get_leaf(rule_1, None)
                    new_G = knowledgebase.Rule(
                        reasoner.dosubst(G.head, subst),
                        [reasoner.dosubst(atom, subst) for atom in new_body],
                    )
                    # G_Head = new_G.head

                    if do_trace and (
                        max_depth.num_nodes <= TRACE_MAX
                        or path_obj.depth <= TRACE_UP_TO_MIN
                    ):
                        print(
                            "("
                            + str(path_obj.depth)
                            + ") Call: "
                            + str(a1)
                            + " (one of "
                            + str(len(G.body))
                            + " subgoals)"
                        )
                        print(
                            "\tRule:"
                            + str(rule_1)
                            + " [Score: "
                            + str(score)
                            + ", "
                            + str(rule_seq)
                            + " of "
                            + str(len(valid_rules))
                            + " matches]"
                        )
                    if path_obj.depth < FALLBACK_DEPTH:
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
                            num_pred,
                            num_var,
                            num_const,
                            do_trace,
                            use_min_score,
                        )
                        # yield ret_val
                        if success:
                            if do_trace and max_depth.num_nodes < TRACE_MAX:
                                print(
                                    "("
                                    + str(path_obj.depth)
                                    + ") Exit: "
                                    + str(bindings)
                                )
                            return success, bindings
                        else:  # added in case the lack of else was causing an issue
                            if do_trace and max_depth.num_nodes < TRACE_MAX:
                                print("(" + str(path_obj.depth) + ") Fail")
                            # continue

                    # no_ans = False
                    # return backwardmainguided(KB, new_G, vars, new_leaf, max_depth, model, guidance_model, depth_list, reasoner_name)
                    # no_ans = False
                    else:
                        success, bindings = backwardmainbasic(
                            KB, new_G, vars, new_leaf, max_depth, depth_list, do_trace
                        )
                        #     # yield ret_val
                        # if ret_val != False:
                        #    print(str(globalCount) + " Exit: " + str(finalRule))
                        if success:
                            if do_trace and max_depth.num_nodes < TRACE_MAX:
                                print(
                                    "("
                                    + str(path_obj.depth)
                                    + ") Exit: "
                                    + str(bindings)
                                )
                            return success, bindings
                        else:
                            if do_trace and max_depth.num_nodes < TRACE_MAX:
                                print("(" + str(path_obj.depth) + ") Fail")
                else:
                    print("ERROR: Should not get here: subst=" + str(subst))
                rule_seq = rule_seq + 1
            # if the code gets here, none of the valid rules worked
            return False, {}
            # if no_ans:
        #     return False
        else:
            return False, {}
    else:  # G.body is empty, should only get here when successful
        if path_obj.depth < max_depth.max:
            max_depth.set(path_obj.depth)
        # yield {vars[i] : G.head.arguments[i] for i in range(len(vars))}
        depth_list.append(path_obj.depth)
        if DEBUG:
            print("Solution: ")
            path_obj.print_rule_path()
            print()
        return True, {vars[i]: G.head.arguments[i] for i in range(len(vars))}


# If this is ever used again, it  needs to be fixed in the same way the regular guided was fixed
def backwardmainguidedalt(
    KB: knowledgebase.KnowledgeBase,
    G: knowledgebase.Rule,
    vars,
    path_obj: knowledgebase.Path,
    max_depth: reasoner.MaxDepth,
    model,
    guidance_model,
    depth_list,
    reasoner_name,
    num_pred,
    num_var,
    num_const,
    do_trace=False,
):
    global globalCount
    global exitGlobalCount
    global goalNumber
    global failFlag
    global callFlag
    global redoFlag
    global keepGoing
    global queryList
    max_depth.num_nodes += 1
    if path_obj.depth > int(max_depth.max * 1.5):  # depth limiter
        return False
    if G.body:
        no_ans = True
        valid_rules = []
        rules_list = []
        i = 0
        for arg in G.body:
            goal_rules = []
            for rule in KB.rule_by_pred[
                arg.predicate
            ]:  # loop to go through rules and check if it unifies w arg before adding to valid_rules
                # standardizes rule
                rule_1 = copy(rule)
                reasoner.standardize(rule_1, path_obj.depth)
                subst = reasoner.unify(arg, rule.head)
                if isinstance(subst, dict):  # if it does, add to valid_rules
                    if reasoner_name == "unity" or reasoner_name == "autoencoder":
                        goal_rules.append(
                            (
                                rule,
                                score_rule_query(
                                    arg, rule, model, guidance_model),
                                i,
                            )
                        )
                    elif reasoner_name == "termwalk":
                        goal_rules.append(
                            (
                                rule,
                                score_rule_query_termwalk(
                                    arg,
                                    rule,
                                    guidance_model,
                                    num_pred,
                                    num_var,
                                    num_const,
                                ),
                                i,
                            )
                        )
                    elif reasoner_name == "chainbased":
                        goal_rules.append(
                            (
                                rule,
                                score_rule_query_chainbased(
                                    arg, rule, guidance_model),
                                i,
                            )
                        )
                    else:
                        raise ValueError
                else:
                    continue
            i += 1
            goal_rules.sort(key=lambda x: x[1], reverse=True)
            rules_list.append(goal_rules)
        rules_list.sort(key=lambda x: x[0][1])
        for i in rules_list:
            valid_rules = valid_rules + i
        failFlag = False
        if valid_rules:  # added this if statement in case valid_rules was left empty
            for best_goal in valid_rules:
                body = deepcopy(G.body)
                a1 = body.pop(best_goal[2])
                score = best_goal[1]
                rule = best_goal[0]
                # #standardizes rule - to do: record the unifier in valid_rules so that we dont have to do the next 3 lines
                rule_1 = copy(rule)
                reasoner.standardize(rule_1, path_obj.depth)
                # #unifies atom and rule head
                subst = reasoner.unify(a1, rule_1.head)
                # #if unification fails, continue on to the next rule
                # If the current atom in our program unifies with our given rule, that means the query should proceed.
                # Unification is determined through previous iterations.
                if not isinstance(
                    subst, bool
                ):  # had to makes sure subst is not boolean to avoid AttributeError: 'bool' object has no attribute 'keys' error;
                    new_body = rule_1.body + body
                    new_leaf = path_obj.get_leaf(rule, None)
                    new_G = knowledgebase.Rule(
                        reasoner.dosubst(G.head, subst),
                        [reasoner.dosubst(atom, subst) for atom in new_body],
                    )
                    G_Head = new_G.head
                    if rule_1 not in queryList:
                        queryList.append(rule_1)
                    if globalCount < 20 and keepGoing == True and subst:
                        queryRule = knowledgebase.Rule(a1, body)
                        if redoFlag == True and failFlag == True:
                            redoAtomQuery = queryList[-1]
                            redoAtomHead = redoAtomQuery.head
                            print(str(globalCount) +
                                  " Redo: " + str(redoAtomHead))
                            redoFlag = False
                            failFlag = False
                        else:
                            print(str(globalCount) + " Call: " + str(rule_1))
                            globalCount += 1
                            print("Goal: " + str(a1))
                            print("Rule: " + str(rule))
                            print("Score: " + str(score))
                            print("Goals to Prove: " + str(new_G))
                            failFlag = False
                        exitGlobalCount += 1
                    else:
                        exitGlobalCount += 1
                    if globalCount >= 100:
                        globalCount = 0
                        keepGoing = False
                    if path_obj.depth <= 3:
                        ret_val = backwardmainguidedalt(
                            KB,
                            new_G,
                            vars,
                            new_leaf,
                            max_depth,
                            model,
                            guidance_model,
                            depth_list,
                            reasoner_name,
                            num_pred,
                            num_var,
                            num_const,
                        )
                        if ret_val:
                            return ret_val
                        else:  # added in case the lack of else was causing an issue
                            continue
                    else:
                        ret_val = backwardmainbasic(
                            KB, new_G, vars, new_leaf, max_depth, depth_list
                        )
                        if ret_val:
                            return ret_val
                else:
                    if globalCount < 20 and keepGoing == True:
                        print(str(globalCount) +
                              " Fail: " + str(queryList[-1]))
                        globalCount += 1
                        redoFlag = True
                        failFlag = True
                    elif globalCount >= 20:
                        globalCount = 0
                        keepGoing = False
                    else:
                        redoFlag = True
                        failFlag = True
                    exitGlobalCount += 1
        else:
            return False
    else:
        if path_obj.depth < max_depth.max:
            max_depth.set(path_obj.depth)
        depth_list.append(path_obj.depth)
        return {vars[i]: G.head.arguments[i] for i in range(len(vars))}


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
    if max_depth.num_nodes > 100000:
        print()  # add new line after dots
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
    if max_depth.num_nodes % 100000 == 0:
        print(".", end="", flush=True)

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
            # if rule_1 not in queryList:
            #    queryList.append(rule_1)
            # if redoFlag == True:
            #    if globalCount < 100:
            #        print("Redo: " + str(rule_1))
            #    redoFlag = False
            # depth_list.append(path_obj.depth)
            reasoner.standardize(rule_1, path_obj.depth)
            # unifies atom and rule head
            subst = reasoner.unify(a1, rule_1.head)
            # if unification fails, continue on to the next rule

            # Generates new rule to pass into next iteration of backwardmainbasic for backward chaining reasoner.
            if isinstance(subst, bool) and not subst:
                # if globalCount < 100:
                #    print("Fail: " + str(rule))
                # redoFlag = True
                continue
            else:
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

        depth_list.append(path_obj.depth)  # added
        return True, {vars[i]: G.head.arguments[i] for i in range(len(vars))}


# uses backward chaining guided reasoning to compute # of nodes visited to reach each query
# returns average number of nodes visited across all queries
def guided(
    queries,
    model,
    guidance_model,
    reasoner_name,
    data,
    kb_file="randomKB.txt",
    alt_select=False,
    do_trace=False,
    use_min_score=True,
    num_pred=10,
    num_var=10,
    num_const=100,
    max_depth_val=10  # changing from 7 to make it possible to finish LUBM queries
):
    global globalCount
    global exitGlobalCount
    # global failFlag
    # global keepGoing
    global timeToExecute
    # the newline='' params below are to keep the output from having extra blank lines in Windows

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
    # i = 0
    for i in range(len(queries)):
        # for i in range(10):
    # for i in [4,11]:
        time_start = process_time()
        query = queries[i]
        print("Query " + str(i) + ": " + str(query))
        i += 1
        path_guide = knowledgebase.Path(query, None, None, 0)
        max_depth_guide = reasoner.MaxDepth(max_depth_val)
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
            alt_select,
            num_pred,
            num_var,
            num_const,
            do_trace,
            use_min_score,
        )
        if not success:
            fail_queries = fail_queries + 1
        # for x in guided_answers:
        #     break
        # # print(max_depth_guide.num_nodes)
        guided_count_total.append(max_depth_guide.num_nodes)
        # print("Nodes Traversed on this query: " + str(max_depth_guide.num_nodes))
        # if failFlag == True:
        #    print(str(exitGlobalCount) + " Fail: "+ str(query))
        # else:
        # If query of string concluded successfully, prints exit message.
        # Accounts for 0-index node count of final node traversal.

        # Old Trace Code!!!
        # exitGlobalCount += 1
        # if failFlag == False:
        #     print(str(exitGlobalCount) + " Exit: " + str(query))
        #     print()
        # else:
        #     print(str(exitGlobalCount) + " Fail: " + str(query))
        #     print()

        time_end = process_time()
        print("Time on this query: " + str(time_end - time_start))
        timeToExecute += time_end - time_start
        exitGlobalCount = 0
        globalCount = 0
        # failFlag = True
        # keepGoing = True
        nodesTraversed = max_depth_guide.num_nodes
        row = [str(query), str(nodesTraversed)]
        writer.writerow(row)
        # min_depths = []
        if depths != []:
            min_dep = min(depths)
        else:
            min_dep = 0
        # # for x in guided_answers:
        #     for x in depths:
        #         min_depths.append(x)
        #         min_dep = min(min_depths)

        nodes = max_depth_guide.num_nodes

        print(i, max_depth_guide.num_nodes, reasoner_name, min_dep)
        if reasoner_name == "unity":
            data_dict.append(
                {
                    "query": i,
                    "unity reasoner": reasoner_name,
                    "unity nodes explored": nodes,
                    "unity min depth": min_dep,
                    "success": success,
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
        print()

    f.close()
    guide_mean_total = sum(guided_count_total) / len(guided_count_total)
    print(f"guided: {guide_mean_total}")
    if fail_queries > 0:
        print(str(fail_queries) + " queries failed")

    data += data_dict
    return guide_mean_total


# uses backward chaining basic reasoning to compute number of nodes visited to reach each query
# returns average number of nodes visited across all queries
def base(queries, reasoner_name, data, kb_file="randomKB.txt", do_trace=False, max_depth_val=10):
    KB = parse_KB_file(kb_file)
    # i = 0
    base_data_dict = []
    base_count_total = []
    for i in range(len(queries)):
        # for i in range(min(10, len(queries))):
        # for i in [14]:
        query = queries[i]
        print("Query " + str(i) + ": " + str(query))
        i += 1
        path_base = knowledgebase.Path(query, None, None, 0)
        max_depth_base = reasoner.MaxDepth(max_depth_val)
        depths = []
        start_time = process_time()
        base_answers = backwardchainbasic(
            KB, path_base, max_depth_base, depths, do_trace
        )
        end_time = process_time()

        base_count_total.append(max_depth_base.num_nodes)

        if depths != []:
            min_dep = min(depths)
        else:
            min_dep = 0

        nodes = max_depth_base.num_nodes

        print(i, max_depth_base.num_nodes, reasoner_name, min_dep)
        print()
        base_data_dict.append(
            {
                "query": i,
                "base reasoner": reasoner_name,
                "base nodes explored": nodes,
                "base min depth": min_dep,
                "time": end_time - start_time,
            }
        )

    base_mean = sum(base_count_total) / len(base_count_total)
    print(f"base: {base_mean}")

    data += base_data_dict
    return base_mean


#  loads pre-trained neural network model, generates test queries,
#  and calls base() and guided() to compare performance of two backward chaining algorithms
if __name__ == "__main__":
    aparser = argparse.ArgumentParser(description="Run the experiment.")
    aparser.add_argument(
        "-q",
        "--generate_queries",
        action="store_true",
        help="Generate a new set of queries.",
    )
    aparser.add_argument(
        "--alt_select",
        action="store_true",
        help="Use alternative control strategy of getting best goal from worst rule.",
    )
    aparser.add_argument("-s", "--standard", action="store_true")
    aparser.add_argument("-u", "--unifier", action="store_true")
    # TODO: everywhere else --kb_path is used instead of --kb
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
    # TODO: maybe change the default to read existing vocab, create a new vocab only by specific argument?
    if args.load_vocab:
        vocab.init_from_vocab(args.vocab_file)
    else:
        print("Initializing vocab from " + args.kb)
        vocab.init_from_kb(parse_KB_file(args.kb))

    input_size = len(vocab.predicates) + (
        (len(vocab.variables) + len(vocab.constants)) * vocab.maxArity
    )
    print("Raw encoding size: " + str(input_size))

    if args.kb:
        if not os.path.isfile(args.kb):
            print("Invalid KB file " + os.path.abspath(args.kb))
            sys.exit(1)
    if args.qfile:
        if not os.path.isfile(args.qfile):
            print("Invalid query file " + os.path.abspath(args.qfile))
            sys.exit(1)
    if args.generate_queries:
        if not os.path.isfile("one_names.csv"):
            print("one_names.csv not found")
            sys.exit(1)
        #        guide = Process(target = guided)
        #        basic = Process(target = base)
        #        guide.start()
        #        basic.start()
        #        guide.join()
        #        basic.join()
        with open("one_names.csv") as a:
            lines = a.readlines()
            with open("uq.txt", "w") as p:
                for line in lines:
                    line = line.strip()
                    p.write(line[: len(line)] + "." + "\n")
        facts_list = parse_KB_file("random_facts.txt").rules
        used_queries = parse_KB_file("uq.txt").rules
        used_queries = deepcopy(used_queries)
        for i in range(len(used_queries)):
            used_queries[i] = used_queries[i].head
        queries = []
        while len(queries) < 100:
            query = reasoner.gen_random_query(facts_list)
            used = False
            for q in used_queries:
                if q == query:
                    used = True
            for q in queries:
                if q == query:
                    used = True
            if not used:
                queries.append(query)
        rules = [knowledgebase.Rule(query, []) for query in queries]
        kbparser.KB_to_txt(knowledgebase.KnowledgeBase(rules), args.qfile)

    # guide = Process(target = guided)
    # basic = Process(target = base)
    # guide.start()
    # basic.start()
    # guide.join()
    # basic.join()
    # with open("one_names.csv") as a:
    #     lines = a.readlines()
    #     with open("uq.txt", "w") as p:
    #         for line in lines:
    #             line = line.strip()
    #             p.write(line[:len(line)] + "." + "\n")
    # facts_list = parse_KB_file("random_facts.txt").rules  # set earlier, not used later
    # used_queries = parse_KB_file("uq.txt").rules
    # used_queries = deepcopy(used_queries)
    # for i in range(len(used_queries)):
    #     used_queries[i] = used_queries[i].head
    # queries = []
    # while len(queries) < 100:
    #     query = reasoner.gen_random_query(facts_list)
    #     used = False
    #     for q in used_queries:
    #         if q == query:
    #             used = True
    #     for q in queries:
    #         if q == query:
    #             used = True
    #     if not used:
    #         queries.append(query)
    # rules = [knowledgebase.Rule(query, []) for query in queries]
    # parser.KB_to_txt(knowledgebase.KnowledgeBase(rules), "test_queries.txt")

    test_queries = kbparser.parse_KB_file(args.qfile).rules
    queries = [query.head for query in test_queries]
    print("START")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device} device")
    data1, data2, data3, data4, data5 = [], [], [], [], []

    # run the selected models on the test queries: standard, unifier, autoencoder, termwalk, and/or chain-based

    use_min_score = not args.no_min_score
    if not use_min_score:
        print("Not using a minimum score for rules")

    print()
    print("WARNING: A better version of the guided reasoner is in pathguidance_imrpoved.py!")
    print()

    # separate booleans for basic reasoner
    if args.standard:
        print("STANDARD")
        base(queries, "standard", data1, args.kb, args.trace)

        with open("standard_data.csv", mode="w", newline="") as file:
            fieldnames = [
                "query",
                "base reasoner",
                "base nodes explored",
                "base min depth",
                "time",
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
        print(f"Embed size: {embed_size}")
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
            args.alt_select,
            args.trace,
            use_min_score,
        )
        print("Time took to execute program: " + str(timeToExecute))
        with open(f"unity_data-{len(vocab.predicates)}-{len(vocab.constants)}-{vocab.maxArity}-{embed_size}.csv", mode="w", newline="") as file:
            fieldnames = [
                "query",
                "unity reasoner",
                "unity nodes explored",
                "unity min depth",
                "success",
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
            args.trace,
            use_min_score,
            args.predicates,
            args.variables,
            args.constants,
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

from knowledgebase import KnowledgeBase, Rule, Path
from vocab import Vocabulary

import time
from time import process_time
from functools import lru_cache
from typing import Any, Literal
from basictypes import Atom, Variable, Constant, Predicate
from copy import deepcopy
import basictypes
import random
from copy import copy

from prints import clear_line, print_progress_bar

# The following constants are imported by mr_back_reasoner.py to ensure consistency
NODE_MAX = 10_000_000  # The maximum number of nodes explored before reasoning is aborted

TRACE_MAX = 150  # The maximum number of reasoning steps that trace information will be output for (per query)
TRACE_UP_TO_MIN = 2  # Level of search shown by trace, even after max nodes is reached


# Alex's comments:
# IMPORTANT MAIN FUNCTIONS
# backwardchain(KnowledgeBase, Path) -> generator of answers:dict
#      path contains the query atom as its node.
# forwardchain(KnowledgeBase) -> None
#      infers all possible facts. Inefficient approach currently
# gen_random_query(KnowledgeBase) -> Atom
#      chooses a random fact, may change some constants to variables
# parse_KB_file(file_path) -> KnowledgeBase
#      generates a knowledge base from a txt file. Follows basic Prolog syntax, but doesn't support all Prolog features.
#      Supports comments with % but must be on separate line


# simple class providing max depth
class MaxDepth:
    def __init__(self, val) -> None:
        self.max = val
        self.num_nodes = 0

    def set(self, val):
        self.max = val


class CachedUnify:

    # @lru_cache(maxsize=None)
    @lru_cache(maxsize=10000)
    def unify_memoized(self, rule_head: Atom, arg: Atom) -> dict[Variable, Any] | Literal[False]:
        return unify(rule_head, arg)


cache = CachedUnify()


# TODO: Verify that clean_atom() and clean_rule() are no longer used. Then delete both
def clean_atom(atom: Atom):
    for i in range(atom.arity):
        if isinstance(atom.arguments[i], Variable):
            atom.arguments[i].name = atom.arguments[i].name[:2]


# takes rule and applies clean_atom to head and body
def clean_rule(rule: Rule):
    clean_atom(rule.head)
    for i in range(len(rule.body)):
        clean_atom(rule.body[i])


def unify(t1: basictypes.Atom, t2: basictypes.Atom):
    """Implements unification algorithm for two atoms.
    Determines whether two logical expressions (Atom objects) can be made identical by applying the
    same subsitution to each.
    Returns false if the terms cannot be unified. Otherwise returns a dictionary of bindings.
    When binding a variable to a different variable, the variable from t1 is the key.

    :param t1: An atom to be unified
    :param t2: An atom to be unified
    :return: A dictionary of bindings (with variable as keys) or False if unify fails
    """

    # Predicate Matching: if the predicates of t1 and t2 are the same and have the same arity.
    if t1.predicate != t2.predicate:
        return False

    # initial list of equations (E) based on the arguments of t1 and t2, ensuring each pair of corresponding arguments will be examined for possible unification.
    E = [[t1.arguments[i], t2.arguments[i]] for i in range(t1.predicate.arity)]
    S = {}

    while E:
        alpha, beta = E.pop()

        if alpha != beta:
            # If both alpha and beta are constants and differ, unification fails
            if isinstance(alpha, Constant) and isinstance(beta, Constant):
                return False
            # If either alpha or beta is a variable, the function attempts to establish a binding between the variable and the other element (either a constant or another variable).
            # This binding is then applied to the remaining equations in E and the substitution set S.
            for var, value in [(alpha, beta), (beta, alpha)]:
                if isinstance(var, Variable):
                    S = {key: value if val ==
                                       var else val for key, val in S.items()}
                    S[var] = value
                    for i, j in [(i, j) for i in range(len(E)) for j in range(2)]:
                        E[i][j] = value if E[i][j] == var else E[i][j]

    # Ensure all keys are variables
    for key in set(S):
        if isinstance(key, Constant):
            raise NotImplementedError
    return S


def old_unify(t1: basictypes.Atom, t2: basictypes.Atom):
    """Implements unification algorithm for 2 input variables.
    Returns false if the terms cannot be unified. Otherwise returns a dictionary of bindings.
    When binding a variable to a different variable, the variable from t1 is the key.

    :param t1: An atom to be unified
    :param t2: An atom to be unified
    :return: A dictionary of bindings (with variable as keys) or False if unify fails
    """
    E = []
    if t1.predicate != t2.predicate:
        return False
    for i in range(t1.predicate.arity):
        E.append([t1.arguments[i], t2.arguments[i]])
    S = {}

    while E:
        term = E.pop()
        alpha = term[0]
        beta = term[1]
        if alpha != beta:
            if isinstance(alpha, Variable):
                for key in S.copy().keys():
                    if S[key] == alpha:
                        S[key] = beta
                    if key == alpha:
                        x = S.pop(key)
                        S[beta] = x
                for i in range(len(E)):
                    for j in range(2):
                        if E[i][j] == alpha:
                            E[i][j] = beta
                S[alpha] = beta
            elif isinstance(beta, Variable):
                for key in S.copy().keys():
                    if S[key] == beta:
                        S[key] = alpha
                    if key == beta:
                        x = S.pop(key)
                        S[alpha] = x
                for i in range(len(E)):
                    for j in range(2):
                        if E[i][j] == beta:
                            E[i][j] = alpha
                S[beta] = alpha
            else:
                return False

    # makes sure all keys are variables
    for key in S.copy().keys():
        if isinstance(key, Constant):
            raise NotImplementedError
            S[S[key]] = key
            S.pop(key)
    return S


# takes subst dictionary and prints the contents in a readable form
def print_subst(a_subst: dict):
    if isinstance(a_subst, bool):
        print("false")
    else:
        for key in a_subst:
            print(f"{key} / {a_subst[key]}")


# returns number of values input generator produces
def gen_count(subst_gen):
    count = 1
    try:
        next(subst_gen)
        count += 1
    except:
        return count


def standardize(rule: Rule, standard: int):
    """ Standardizes apart all variable names of rule. Assumes that the standard is a unique
    number that won't appear anywhere else on the same inference path. Is typically
    assumed to be the depth of the current search. Modifies the rule
    by appending a unique identifier to the names of all variables. Note, this WILL
    change the original rule argument.

    :param rule: The rule to standardize
    :param standard: An integer, usually the depth of the current inference step
    """
    for arg in rule.head.arguments:
        if isinstance(arg, Variable):
            arg.name = arg.name + str(standard)
    for atom in rule.body:
        for arg in atom.arguments:
            if isinstance(arg, Variable):
                arg.name = arg.name + str(standard)


# TODO: eventually replace all dosubst[_] refs with the version in Atom
# modified atom object by applying susbt dictionary to its arguments
def dosubst_(atom: Atom, subst: dict):
    for i in range(atom.arity):
        if atom.arguments[i] in subst.keys():
            atom.arguments[i] = subst[atom.arguments[i]]


# Returns a copy of atom object with subst dictionary applied
def dosubst(atom: Atom, subst: dict):
    return Atom(
        atom.predicate,
        [subst.get(arg, arg) for arg in atom.arguments]
    )


# calls the 2 preceding methods
def sub_rule(rule: Rule, subst: dict) -> Rule:
    """
    Returns a new rule that is the resutl of apply subst to rule
    :param rule:
    :param subst:
    :return:
    """
    return Rule(rule.head.dosubst(subst), [x.dosubst(subst) for x in rule.body])


# attempts to unify 2 rule inputs; returns true or false depending on whether they can be unified
def unify_rules_equal(rule1: Rule, rule2: Rule):
    rule2 = copy(rule2)
    rule1 = copy(rule1)
    standardize(rule2, 1)
    subst = unify(rule1.head, rule2.head)
    if isinstance(subst, bool) and not subst:
        return False
    else:
        for i in range(len(rule2.body)):
            rule2.body[i] = dosubst(rule2.body[i], subst)
        dosubst_(rule2.head, subst)
        for i in range(len(rule1.body)):
            rule1.body[i] = dosubst(rule1.body[i], subst)
        dosubst_(rule1.head, subst)
        if rule1 == rule2:
            return True
        else:
            return False


# checks whether 2 atoms are equal by comparing predicates and arguments
def eq_atoms_forward_chaining(atom1: Atom, atom2: Atom):
    if atom1.predicate != atom2.predicate:
        return False
    else:
        for i in range(atom1.arity):
            if isinstance(atom1.arguments[i], Constant):
                if atom1.arguments[i] != atom2.arguments[i]:
                    return False
            elif isinstance(atom2.arguments[i], Constant):
                if atom1.arguments[i] != atom2.arguments[i]:
                    return False
    return True


# takes rule with variables and returns all possible ground instances of rule
# likely not useful (gets big very fast!)
def ground_rule(rule: Rule, ground_rules: set, vars: set):
    if not vars:
        ground_rules.add(rule)
    else:
        for var in vars:
            for i in range(99):
                ground_rule(
                    sub_rule(rule, {var: Constant("a" + str(i))}),
                    ground_rules,
                    vars.symmetric_difference({var}),
                )


def trysubst(rule, KB, new, cu: CachedUnify | None = None):
    """
    Tries to substitute variables in rule with constants to match facts;
    if it succeeds and the head is new, adds the head to the "new" set
    :param rule:
    :param KB:
    :param new:
    :param cu:
    :return:
    """
    # Check if the rule is already in the new set
    if rule in new:
        return

    # Check against existing rules with an empty body (i.e. facts)
    for o_rule in (r for r in KB.rules if not r.body):
        if eq_atoms_forward_chaining(rule.head, o_rule.head):
            return

    # If the rule has no body, add it to the new set
    if not rule.body:
        new.add(rule)
        return

    # Rule has a body, make a copy to work with
    new_rule = copy(rule)

    # Pop the last element from the body
    arg = new_rule.body.pop()

    # Iterate over old rules for the predicate of the argument
    for old_rule in KB.try_index_pred(arg.predicate):
        if not old_rule.body:
            # Unify the heads
            unification = unify(old_rule.head, arg) if cu is None else cu.unify_memoized(
                old_rule.head, arg)
            if not isinstance(unification, bool):
                # Substitute and recursively call trysubst
                rule_try = sub_rule(new_rule, unification)
                trysubst(rule_try, KB, new, cu)


def standardize_fact(fact: Atom, standard: int) -> Atom:
    """
    Returns a copy of the fact that replaces variables with unused variables.
    :param fact:
    :param standard: number used to make var umique. Usually depth of search
    :return:
    """
    std_fact: Atom = deepcopy(fact)
    for arg in std_fact.arguments:
        if isinstance(arg, Variable):
            arg.name = arg.name + str(standard)
    return std_fact


def apply_rule_step(rule: Rule, facts: dict[Predicate, list[Atom]],
                    cu: CachedUnify) -> list[Atom]:
    """ Applies one forward-chain step for the rule. For each combination of facts that matches
    the body, will return one corresponding head of the rule (after applying the matching
    substitutions). Is typically used after matching one of the body goals to a fact from the
    last iteration, and the passing in the rule without that goal.
    :param rule: A rule to apply (after adjusting for first match). Can be ground or not.
    :param facts: A dictionary of known facts keyed by predicate
    :param cu: A class to keep track of memoized unify calls
    :return:
    """
    new_facts = []
    if not rule.body:  # empty body is success
        return [rule.head]
    g = rule.body.pop()
    # Note, if we standardize g, then the returned substitutions will not be useful
    # g = standardize_fact(g, depth)   # standardize g so don't have to standardize facts when unifying below
    if g.predicate in facts:
        same_facts_pred = facts[g.predicate]
    else:
        same_facts_pred = []
    for f in same_facts_pred:
        result = cu.unify_memoized(f, g)
        if not isinstance(result, bool):
            if not rule.body:  # no goals remain, we have an answer
                # NOTE: rather than returning, we need to remember the answer, AND keep looking for more
                new_facts.append(rule.head.dosubst(result))
                # return result
            else:  # apply the substitution and process the remaining goals
                new_rule = sub_rule(rule, result)
                # new_goals = [new_g.dosubst(result) for new_g in goals]
                next_facts = apply_rule_step(new_rule, facts, cu)
                new_facts.extend(next_facts)
                # if not isinstance(next_subst, bool):
                #     new_facts.append(new_rule.head.dosubst(next_subst))
                #     # result.update(next_subst)
                # return result
    return new_facts
    # return False            # if it gets here, tried every possible fact and subsequent substitution with no success


def forwardchain(KB: KnowledgeBase, vocab: Vocabulary) -> (set[Rule], int):
    """
    Forward chaining algorithm: performs on input KB bu applying rules to generate new facts
    until no new facts can be generated
    :param KB:
    :return:
    """
    # KB = deepcopy(K)
    facts: set[Rule] = set()
    cu = CachedUnify()
    depth = 1

    all_facts = {}
    for rule in KB.rules:
        if not rule.body:
            fact = rule.head
            if fact.predicate in all_facts:
                all_facts[fact.predicate].append(fact)
            else:
                all_facts[fact.predicate] = [fact]

    all_rules = [rule for rule in KB.rules if rule.body]
    last_facts = [rule.head for rule in KB.rules if not rule.body]

    while True:
        print(f"Forward chaining depth: " + str(depth))
        new = set()

        for rule in all_rules:
            rule = deepcopy(rule)  # avoid clobbering the rule
            # standardize the rule to avoid standardizing facts in match_sub()
            # by standardizing here, we avoid re-standardizing in the inner-most loop (and very long var names)
            standardize(rule, depth)
            for i, body_predicate in enumerate(rule.body):
                for candidate in last_facts:
                    # since we are standardizing the rules, there's no need to standardize the facts too
                    # goal_seq = 1
                    # candidate = standardize_fact(candidate, goal_seq)

                    unification = cu.unify_memoized(rule.body[i], candidate)
                    if not isinstance(unification, bool):
                        # if candidate.predicate.name == "ancestor":
                        #     if candidate.arguments[0] == Constant("cersei_lannister"):
                        #         print("Trying rules with new fact: " + str(candidate))
                        rule_try = sub_rule(rule, unification)  # apply the subst to the rule
                        # Note, rule heads that aren't ground can be useful for subsequent reasoning
                        # Also, the head may become ground after subsequent substitutions below
                        # if rule_try.head.is_ground():
                        if rule_try.head.predicate in all_facts:
                            same_pred_facts = all_facts[rule_try.head.predicate]
                        else:
                            same_pred_facts = []
                        if rule_try.head not in new and rule_try.head not in same_pred_facts:
                            rule_try.body.pop(i)
                            rule_facts = apply_rule_step(rule_try, all_facts, cu)
                            for rf in rule_facts:
                                # TODO: need to pass in vocab and sanitize atom to ensure finite set of variables
                                rf = vocab.sanitize_atom(rf)
                                if rf not in same_pred_facts and rf not in new:
                                    new.add(rf)
                                # if not isinstance(match_sub, bool):
                                #     head_result = rule_try.head.dosubst(match_sub)
                                #     if head_result not in same_pred_facts:
                                #         new.add(rule_try.head.dosubst(match_sub))
                                print(f"\rNew facts: " + str(len(new)), end="")

        # for rule in KB.rules:
        #     for i, body_predicate in enumerate(rule.body):
        #         old_rules = last_iter.try_index_pred(body_predicate.predicate)
        #         non_empty_rules = [
        #             old_rule for old_rule in old_rules if not old_rule.body]
        #         for old_rule in non_empty_rules:
        #             r = copy(rule)
        #             standardize(r, standard)
        #             standard += 1
        #             unification = cu.unify_memoized(old_rule.head, r.body[i])
        #             if not isinstance(unification, bool):
        #                 rule_try = sub_rule(r, unification)     # apply the subst to the rule
        #                 if rule_try.head.is_ground():
        #                     if rule_try.head not in new and rule_try.head not in KB.rule_by_pred:
        #                         rule_try.body.pop(i)
        #                         trysubst(rule_try, KB, new)     # NOTE: this will update new
        #                         if found_match:
        #                             facts.add(rule_try.head)
        #                             print(f"\rNew facts: " + str(len(new)), end="")
        if not new:
            break
        for fact in new:
            # update the working set of all_facts proven so far
            if fact.predicate in all_facts:
                all_facts[fact.predicate].append(fact)
            else:
                all_facts[fact.predicate] = [fact]

            # update the complete set of new facts
            facts.add(Rule(fact, []))

        last_facts = new

        # last_iter = KnowledgeBase(list(new))
        # facts.update(new)
        depth += 1
        print()
    print()

    return facts, depth - 1


# generates random query by selecting fact from KB and replacing some constants with variables
def gen_random_query(facts: list[Rule]) -> Atom:
    """
    This method is DEPRECATED. It assumes that the vocab has variables X0,..X9
    :param facts:
    :return:
    """
    query = deepcopy(random.choice(facts))
    # print(str(query.head.arguments) + " -> ", end="")

    # generate list of variables
    nums = [Variable("X" + str(j)) for j in range(9)]
    # remove variables that are already in query from list
    nums = [item for item in nums if item not in query.head.arguments]
    for i in range(len(query.head.arguments)):
        if random.random() < 0.8 ** ((10 - len(nums)) + len(query.head.arguments) - i):
            if isinstance(query.head.arguments[i], Constant):
                # choose random variable and remove it from list
                rand = random.choice(nums)
                nums.remove(rand)
                # replace all occurrences of constant with variable
                const = query.head.arguments[i]
                query.head.arguments = [
                    rand if x == const else x for x in query.head.arguments
                ]

    # print(query.head.arguments, end="\n")
    return query.head


def gen_random_query_vocab(facts: list[Rule], vocab: Vocabulary) -> Atom:
    """
    Generates a single random query by selecting fact from KB and replacing some constants with variables

    :param facts:
    :param vocab: Determines what variables are avaible to use in the query.
    :return:
    """
    query = deepcopy(random.choice(facts))
    # print(str(query.head.arguments) + " -> ", end="")

    # generate list of variables
    nums = vocab.variables
    # remove variables that are already in query from list
    nums = [item for item in nums if item not in query.head.arguments]
    for i in range(len(query.head.arguments)):
        if random.random() < 0.8 ** ((10 - len(nums)) + len(query.head.arguments) - i):
            if isinstance(query.head.arguments[i], Constant):
                # choose random variable and remove it from list
                rand = random.choice(nums)
                nums.remove(rand)
                # replace all occurrences of constant with variable
                const = query.head.arguments[i]
                query.head.arguments = [
                    rand if x == const else x for x in query.head.arguments
                ]

    # print(query.head.arguments, end="\n")
    return query.head


class BackChainReasoner:
    """ A class to represent a customizable reasoner with machine-learning supported meta-reasoning.
    """
    kb: KnowledgeBase
    vocab = Vocabulary()
    max_depth: int
    reasoner_name: str
    do_trace: bool
    print_solution: bool = False
    trace_file = "trace_log.txt"

    num_nodes = 0
    std_seq = 0  # TODO: delete this?

    __depth = 0  # used by MetaBackChainReasoner in the goal/rule selector methods

    def __init__(self, kb: KnowledgeBase, vocab: Vocabulary,
                 max_depth=15, do_trace=False, print_solution=False):
        """ Initializes a meta-reasoning class.
        :param kb: The knowledge base
        :param vocab: The vocabulary of the KB
        :param max_depth: The deepest the reasoner will search before aborting
        :param do_trace: Boolean to determine if trace info should be output.

        """

        self.kb = kb
        self.vocab = vocab
        self.do_trace = do_trace
        self.max_depth = max_depth
        self.print_solution = print_solution

    def query(self, goals: list[Atom]):
        """Execute a query using a standard backward-chaining reasoner.

        :param goals: A list of atoms that constitute the query
        """

        path_obj = Path(None, None, None, 0, goals)
        vars = set()
        for subg in goals:
            for arg in subg.arguments:
                if isinstance(arg, Variable):
                    vars.add(arg)
        vars = list(vars)
        G = Rule(Atom(Predicate(len(vars), "yes"), copy(vars)), goals)

        self.num_nodes = 0
        self.std_seq = 0  # reset the seq number used to standardize variables apart
        t = process_time()

        success, bindings, path_obj = self.query_helper(G, set(vars), path_obj, t)
        clear_line()
        if not success:
            print("Query failed!!!")
            # open(self.trace_file, 'a').write("Query failed!!!\n")
        elif self.print_solution:
            print("Solution: ")
            path_obj.print_rule_path()
            print()
        return success, bindings, path_obj

    def query_helper(self,
                     G: Rule,
                     vars: set[Variable],
                     path_obj: Path,
                     start_time=0.0
                     ):
        """Execute one step of a query using a standard backward-chaining reasoner. The KB is
        configured through the class initializer. """

        # made max_depth more accurate by removing 1.5* factor
        if path_obj.depth > self.max_depth or self.num_nodes >= NODE_MAX:
            return False, {}, None

        # if max_depth.num_nodes % 1000 == 1:
        #     diff = process_time() - start_time + 0.0001
        #     clear_line()
        #     print("\r", int(max_depth.num_nodes / diff), '\t',
        #           max_depth.num_nodes, end="", flush=True)

        if ((self.num_nodes % 5000 == 1 or (self.kb.length > 250 and self.num_nodes % 1000 == 1)) and
                (not self.do_trace or self.num_nodes > TRACE_MAX)):
            diff = process_time() - start_time
            print_progress_bar(self.num_nodes, NODE_MAX,
                               shown='percent',
                               suffix=f'to max depth ({int(self.num_nodes / diff) if diff > 0 else "-"} nps)',
                               length=25)

        self.num_nodes += 1
        # self.set_depth(path_obj.depth)     # sets the depth for use by standardize()

        if G.body:

            # a1 = G.body.pop(0)
            a1 = G.body[0]  # don't want to modify the list

            no_ans = True

            # bug fix for key errors on unoptimized KBs
            if a1.predicate not in self.kb.rule_by_pred:
                # no rules for this predicate; skip it
                return False, {}, None

            # for each rule in the KB that could potentially satisfy the atom (based on matching predicates)
            valid_rules = self.match_single_goal(a1, path_obj.depth)

            if valid_rules:  # added this if statement in case valid_rules was left empty
                rule_seq = 0
                for best_goal in valid_rules:

                    next_rule, next_subst = best_goal

                    body = copy(G.body)
                    body.pop(0)

                    # # If the current atom in our program unifies with our given rule, that means the query should proceed.
                    # # Unification is determined through previous iterations.
                    # if isinstance(next_subst, bool):  # had to makes sure subst is not boolean to avoid AttributeError: 'bool' object has no attribute 'keys' error;
                    #     continue

                    new_body = next_rule.body + body

                    new_leaf = path_obj.make_child(a1, next_rule, new_body, len(next_rule.body))
                    new_g = Rule(
                        dosubst(G.head, next_subst),
                        [dosubst(atom, next_subst) for atom in new_body],
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

                    if self.do_trace and (
                            self.num_nodes <= TRACE_MAX
                            or path_obj.depth <= TRACE_UP_TO_MIN
                    ):
                        print(
                            "(" + str(path_obj.depth)
                            + ") Call: " + str(a1)
                            + " (one of " + str(len(G.body)) + " subgoals)"
                        )
                        print(
                            "\tRule:" + str(next_rule)
                            + " [" + str(rule_seq + 1)
                            + " of " + str(len(valid_rules)) + " matches]"
                        )

                    success, bindings, final_path = self.query_helper(
                        new_g,
                        vars,
                        new_leaf,
                        start_time,
                    )

                    if success:
                        if self.do_trace and self.num_nodes < TRACE_MAX:
                            print(
                                "(" + str(path_obj.depth) + ") Exit: " + str(bindings)
                            )

                        return success, bindings, final_path
                    else:
                        if self.do_trace and self.num_nodes < TRACE_MAX:
                            print("(" + str(path_obj.depth) + ") Fail")
                    rule_seq += 1

            # if the code gets here, none of the valid rules worked
            return False, {}, None
        else:  # G.body is empty, should only get here when successful

            return True, {list(vars)[i]: G.head.arguments[i] for i in range(len(vars))}, path_obj

    def set_depth(self, depth):
        self.__depth = depth

    def get_depth(self):
        return self.__depth

    # TODO: To best work with standardize apart, vocab should create 0 to k versions of all variables
    # def standardize(self, rule):
    #     # has_vars = False
    #     for arg in rule.head.arguments:
    #         if isinstance(arg, Variable):
    #             arg.name = arg.name + str(self.__depth)
    #             # arg.name = arg.name + str(self.std_seq)
    #             # has_vars = True
    #     for atom in rule.body:
    #         for arg in atom.arguments:
    #             if isinstance(arg, Variable):
    #                 arg.name = arg.name + str(self.__depth)
    #                 # arg.name = arg.name + str(self.std_seq)
    #                 # has_vars = True
    #     # if has_vars:
    #     #     self.std_seq += 1

    @lru_cache(maxsize=10000)
    def match_single_goal(self, goal: Atom, depth: int) -> list[tuple[Rule, dict | bool]]:
        """ Finds all matching rules for a single goal. This helper method allows the calls to be
        cached, which cannot be done with a list parameter. Note, the depth parameter is essential
        to ensure that the rules returned by the cache have been standardized apart in a
        consistent way so not to conflict with other variables introduced on the same path. """

        valid_rules = []
        if goal.predicate not in self.kb.rule_by_pred:
            print(f"Predicate '{goal.predicate}' not found in the knowledge base")

        for rule in self.kb.rule_by_pred[goal.predicate]:
            rule_1: Rule = copy(rule)
            standardize(rule_1, depth)
            subst = cache.unify_memoized(goal, rule_1.head)
            # subst = reasoner.old_unify(rule_1.head, goal)

            if not isinstance(subst, dict):
                continue

            addition = (rule_1, subst)
            valid_rules.append(addition)
        return valid_rules


if __name__ == "__main__":
    pass

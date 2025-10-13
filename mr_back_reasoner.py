import kbparser
import knowledgebase
import reasoner
import kbencoder
import nnreasoner
import nnunifier
from basictypes import Atom, Predicate, Variable
from vocab import Vocabulary
from reasoner import BackChainReasoner
from reasoner import NODE_MAX, TRACE_MAX, TRACE_UP_TO_MIN
from reasoner import cache
import time
from time import process_time
import os
from copy import copy
from types import FunctionType
from prints import clear_line, print_progress_bar
from functools import lru_cache
from embedmodel import EmbedModel, UnifierEmbed

# need this to fix interrupt issues, must be before any scipy is imported
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import torch

# When this option is used, a rule must have at least the score to be evaluated
MIN_RULE_SCORE = 0.01
# MIN_RULE_SCORE = 0.001
# FALLBACK_DEPTH = 3  # The depth at which the meta-reasoner switches to the standard reasoner
FALLBACK_DEPTH = 5

timeToExecute = 0

device = "cuda" if torch.cuda.is_available() else "cpu"


class MetaBackChainReasoner(BackChainReasoner):
    """ A class to represent a customizable reasoner with machine-learning supported meta-reasoning.
    """
    embed_model: EmbedModel | None
    embed_size: int
    guidance_model: nnreasoner.NeuralNet
    goal_select_func: FunctionType
    rule_select_func: FunctionType
    use_fallback: bool = False

    # this member var is used to pass information between the goal and rule selectors
    working_rules: []


    def __init__(self, kb: knowledgebase.KnowledgeBase, vocab: Vocabulary,
                embed_model: EmbedModel | None, guidance_model: nnreasoner.NeuralNet,
                goal_select_func: FunctionType, rule_select_func: FunctionType,
                max_depth=15, embed_size=50, do_trace=False, print_solution=False):
        """ Initializes a meta-reasoning class.
        :param kb: The knowledge base
        :param vocab: The vocabulary of the KB
        :param embed_model: Embedding model
        :param guidance_model: Reasoning model
        :param goal_select_func: A function for selecting which goal(s) to pursue
        :param rule_select_func: A function for selecting (and ordering) which rules to apply
        :param max_depth: The deepest the reasoner will search before aborting
        :param do_trace: Boolean to determine if trace info should be output.
        :param: print_solution: Boolean to determine if steps of solution should be output to the console
        :param embed_size: Embedding size that results from the embed_model
        """

        super().__init__(kb, vocab, max_depth, do_trace, print_solution)
        # self.kb = kb
        # self.vocab = vocab
        self.embed_model = embed_model
        self.guidance_model = guidance_model
        self.goal_select_func = goal_select_func
        if goal_select_func == MetaBackChainReasoner.all_goals_selector:
            self.use_fallback = True         # the all goals selector should be used with a fallback depth!
            print("Enabled fallback reasoning!")
        else:
            self.use_fallback = False
        self.rule_select_func = rule_select_func
        # self.do_trace = do_trace
        # self.max_depth = max_depth
        self.embed_size = embed_size

    def query(self, goals: list[Atom]):
        """Execute a query using the guided reasoner. Uses backward chaining code. Checks depth,
        evaluates body of G, scores each Atom, sorts scores in descending order, checks each rule,
        tries to unify rules with queries

        :param goals: A list of atoms that constitute the query
        """

        path_obj = knowledgebase.Path(None, None, None, 0, goals)
        vars = set()
        for subg in goals:
            for arg in subg.arguments:
                if isinstance(arg, Variable):
                    vars.add(arg)
        vars = list(vars)
        G = knowledgebase.Rule(
            Atom(Predicate(len(vars), "yes"), copy(vars)), goals)

        self.num_nodes = 0
        self.std_seq = 0       # reset the seq number used to standardize variables apart
        t = process_time()

           # if alt_select:
        #     success, bindings = backwardmainguidedalt(
        #         KB,
        #         G,
        #         set(vars),
        #         path_obj,
        #         max_depth,
        #         model,
        #         guidance_model,
        #         depth_list,
        #         reasoner_name,
        #         trace_file,
        #         use_min_score,
        #         goal_pruning,
        #         t
        #     )
        # else:
        success, bindings, path_obj = self.query_helper(
            G, set(vars),path_obj, t)
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
        G: knowledgebase.Rule,
        vars: set[Variable],
        path_obj: knowledgebase.Path,
        start_time=0.0
    ):
        """Execute one step of a query using the guided reasoner. The KB, embedding, and reasoning
        models are configured through the class initializer. """
        if self.use_fallback and path_obj.depth >= FALLBACK_DEPTH:
            # print("Using Fallback!")
            success, bindings, final_path = super().query_helper(G, vars, path_obj, start_time)
            return success, bindings, final_path

        # made max_depth more accurate by removing 1.5* factor
        if path_obj.depth > self.max_depth or self.num_nodes >= NODE_MAX:
            return False, {}, None

        # if max_depth.num_nodes % 1000 == 1:
        #     diff = process_time() - start_time + 0.0001
        #     clear_line()
        #     print("\r", int(max_depth.num_nodes / diff), '\t',
        #           max_depth.num_nodes, end="", flush=True)

        if ( (self.num_nodes % 5000 == 1 or (self.kb.length > 250 and self.num_nodes % 1000 == 1)) and
            (not self.do_trace or self.num_nodes > TRACE_MAX) ):
            diff = process_time() - start_time
            print_progress_bar(self.num_nodes, NODE_MAX,
                               shown='percent', suffix=f'to max depth ({int(self.num_nodes / diff) if diff > 0 else "-"} nps)', length=25)

        self.num_nodes += 1
        self.working_rules = []       # reset the set of working rules, used by some selector functions
        self.set_depth(path_obj.depth)     # sets the depth for use by standardize()

        if G.body:
            valid_rules: list[tuple[knowledgebase.Rule, float, Atom, dict | bool]] = []

            # The goal_select_func should return a subset of the goals and their corresponding position numbers
            chosen_goals, goal_seqs = self.goal_select_func(self, G.body)

            valid_rules = self.rule_select_func(self, chosen_goals)

            # with open(self.trace_file, 'a') as f:
            #    for r in valid_rules:
            #        best_rule, best_score, a, best_subst = r
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
            #        f.write(str(best_rule.head)+" :- " + str(best_rule.body) + ", " +
            #                str(best_score) + ", " + str(a) + ", " + str(best_subst))
            #        f.write("\n")
            #    f.write("\n\n\n")

            if valid_rules:  # added this if statement in case valid_rules was left empty

                for rule_seq, best_goal in enumerate(valid_rules):
                    # body = copy(G.body)  # copy the subgoals and pop matching goal, this will be used with the rule's body later
                    best_rule, best_score, a1, best_subst = best_goal
                    # print('bod:', body)
                    # a1 = G.body[goal_seq]
                    body = list(set(G.body) - {a1})

                    # If the current atom in our program unifies with our given rule, that means the query should proceed.
                    # Unification is determined through previous iterations.
                    if isinstance(
                        best_subst, bool
                    ):  # had to makes sure subst is not boolean to avoid AttributeError: 'bool' object has no attribute 'keys' error;
                        continue

                    new_body = best_rule.body + body
                    # new_leaf = path_obj.get_leaf(best_rule, None)
                    new_leaf = path_obj.make_child(a1, best_rule, new_body, len(best_rule.body))
                    new_g = knowledgebase.Rule(
                        reasoner.dosubst(G.head, best_subst),
                        [reasoner.dosubst(atom, best_subst)
                         for atom in new_body],
                    )

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
                            "\tRule:" + str(best_rule)
                            + " [Score: " + str(best_score) + ", " + str(rule_seq)
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

            # if the code gets here, none of the valid rules worked
            return False, {}, None
        else:  # G.body is empty, should only get here when successful

            return True, {list(vars)[i]: G.head.arguments[i] for i in range(len(vars))}, path_obj

    def score_rule_query(self, query, rule) -> float:
        """Evaluates query and rule and returns a score.

        :param query: A subgoal (an atom) to evaluate
        :param rule: A rule that could be used to prove the goal
        :return: A score (>=0, <=1) of the likelihood that the rule will eventually lead to a proof
        """

        # TODO: Is there any way to make sure the embed and scoring models are connected in GPU calculations?

        self.guidance_model.eval()      # make sure guidance model is in test model
        with torch.no_grad():              # will this make a difference in speed?
            embedding = self.embed_model.get_goal_rule_embed(query, rule)

            # changed get_score() to accept torch tensors (it was converting  numpy to torch anyway)
            # score = nnreasoner.get_score(embedding.numpy(), self.guidance_model)
            score = nnreasoner.get_score(embedding, self.guidance_model)
            return score

        # self.embed_model.eval()
        # with torch.no_grad():
        #     query = {query: rule}
        #     query = kbencoder.one_hot_encode_query(query, self.vocab)
        #
        #     # Extract tensors outside the loop
        #     atom = self.embed_model(torch.FloatTensor(query[0]).to(
        #         device)).cpu().detach().numpy()
        #     rule_head = self.embed_model(torch.FloatTensor(
        #         query[1]).to(device)).cpu().detach().numpy()
        #     args = torch.zeros(self.embed_size).to(device)
        #
        #     for arg in query[2]:
        #         arg = self.embed_model(torch.FloatTensor(arg).to(device))
        #         args += arg
        #
        #     args = args.cpu().detach().numpy()
        #
        #     # Convert NumPy arrays to PyTorch tensors
        #     atom = torch.FloatTensor(atom)
        #     rule_head = torch.FloatTensor(rule_head)
        #     args = torch.FloatTensor(args)
        #
        #     # Concatenate tensors using torch.cat
        #     embedding = torch.cat([atom, rule_head, args])
        #
        #     # changed get_score() to accept toch tensors (it was converting  numpy to torch anyway)
        #     # score = nnreasoner.get_score(embedding.numpy(), self.guidance_model)
        #     score = nnreasoner.get_score(embedding, self.guidance_model)
        #     return score

    # Note: these are for reference. Should be obsolete due to new, generalized score_rule_query
    # def score_rule_query_termwalk(
    #         query,
    #         rule,
    #         guidance_model,
    #         num_pred: int = 10,
    #         num_var: int = 10,
    #         num_const: int = 100,
    # ):
    #     with torch.no_grad():
    #         query_vec = termwalk.termwalk_representation(
    #             query, num_pred, num_var, num_const
    #         )
    #         rule_vec = termwalk.termwalk_representation(
    #             rule, num_pred, num_var, num_const)
    #         embedding = np.concatenate([query_vec, rule_vec])
    #         score = nnreasoner.get_score(embedding, guidance_model)
    #         return score
    #
    # def score_rule_query_chainbased(query, rule, guidance_model):
    #     with torch.no_grad():
    #         query_vec = chainbased.represent_pattern(query, 20)
    #         rule_vec = chainbased.represent_pattern(rule, 20)
    #         embedding = np.concatenate([query_vec, rule_vec])
    #         score = nnreasoner.get_score(embedding, guidance_model)
    #         return score

    @lru_cache(maxsize=10000)
    # @lru_cache(maxsize=None)
    def match_and_score_single_goal(self, goal:Atom, depth:int) -> list[tuple[knowledgebase.Rule, float, Atom, dict | bool]]:
        """ Finds all matching rules for a single goal. This helper method allows the calls to be
        cached, which cannot be done with a list parameter. Note, the depth parameter is essential
        to ensure that the rules returned by the cache have been standardized apart in a
        consistent way so not to conflict with other variables introduced on the same path. """

        # TODO: consider adding the knowledge base and scoring model to the parameters, to avoid the
        # cache returning wrong results if we change either
        valid_rules = []

        if goal.predicate not in self.kb.rule_by_pred:
            return valid_rules

        for rule in self.kb.rule_by_pred[goal.predicate]:
            rule_1: knowledgebase.Rule = copy(rule)
            reasoner.standardize(rule_1, depth)
            subst = cache.unify_memoized(goal, rule_1.head)
            # subst = reasoner.old_unify(rule_1.head, goal)

            if not isinstance(subst, dict):
                continue

            score = self.score_rule_query(goal, rule)
            # addition = (rule_1, score, i, subst)
            addition = (rule_1, score, goal, subst)
            valid_rules.append(addition)
        return valid_rules

    def match_and_score_rules(self, all_goals: list[Atom], depth:int) -> list[tuple[knowledgebase.Rule, float, Atom, dict | bool]] :
        """ Goes through each rule in the Knowledge Base, determines which ones unify to the goals,
        and applies the scoring function to each matching rule.
        """
        valid_rules: list[tuple[knowledgebase.Rule, float, Atom, dict | bool]] = []

        for i, arg in enumerate(all_goals):
            # if not isinstance(arg, Atom):
            #     print("Bad atom: " + str(arg) + " from " + str(all_goals))
            valid_rules.extend(self.match_and_score_single_goal(arg, depth))

        return valid_rules

    # *******************************************************************
    # Here begin the functions used to specify the control mechanism of the algorithm
    # Control is specified by a goal_selector and a rule_selector
    # 1) goal_selectprs have form [name]_goal_selector, take a list of Atoms as input
    #    and return a list of Atoms
    # 2) rule_selectors have form [name]_rule_selector, take a list of Atoms as input
    #    and return a list of rule tuples (rule, score, goal, bindings)

    def min_goal_selector(self, all_goals:list[Atom]):
        """ Finds all the rules that match each goal, and tracks the maximum scoring rule
        for each goal. Chooses the goal with the lowest maximum score and sets working_rules
        to only the rules for that goal.
        :param all_goals:
        :return:
        """

        goal_selection: dict[int, list[tuple[knowledgebase.Rule, float, Atom, dict | bool]]] = {}
        score_list: list = [None] * len(all_goals)

        # valid_rules = self.match_and_score_rules(all_goals)

        for i, arg in enumerate(all_goals):
            if not isinstance(arg, Atom):
                print("Bad atom: " + str(arg) + " from " + str(all_goals))
            goal_selection[i] = self.match_and_score_single_goal(arg, self.get_depth())

            for (r, score, g, subst) in goal_selection[i]:
                score_list[i] = max(score, score_list[i]) if score_list[i] else score

        min_goal = score_list.index(min(score_list, key=lambda x: x if x else float("inf")))
        if goal_selection:           # set the working rules to just be those that match the selected goal
            self.working_rules = goal_selection[min_goal]
        else:
            self.working_rules = []
        return [all_goals[min_goal]], [min_goal]

    def all_goals_selector(self, all_goals):
        """ Returns the full set of goals in the order given. This is used when we want
        to consider the rules for all goals, as opposed to only a selected goal.
        """

        return all_goals, range(len(all_goals))

    def max_rule_selector(self, goals):
        """ Finds all the rules that match each of the goals, and returns them in
        descending order of score. If working_rules is not empty, will reuse the rules
        found there, so it can leverage the work done by min_goal_selector() and
        similar methods."""
        if not self.working_rules:
            valid_rules = self.match_and_score_rules(goals, self.get_depth())
        else:
            valid_rules = self.working_rules
        valid_rules.sort(key=lambda x: x[1], reverse=True)
        return valid_rules

    def max_rule_with_min_scoring_selector(self, goals):
        """ Like max_rule_selector, will return all rules that match any of the goals,
        but will remove any with scores that are under the MIN_RULE_SCORE threshold.
         """
        valid_rules = self.max_rule_selector(goals)
        for i in range(len(valid_rules), 0, -1):
            r_tuple = valid_rules[i]
            if r_tuple[1] <  MIN_RULE_SCORE:
                valid_rules.pop(i)
            else:             # since rules are sorted, we can stop searching once we find a high enough one
                break

        return valid_rules


# uses backward chaining guided reasoning to compute # of nodes visited to reach each query
# returns average number of nodes visited across all queries
def guided(
    queries: list[Atom],
    mr_reasoner: reasoner.BackChainReasoner,
    data: list,
    reasoner_name = "unity",
):
    global globalCount
    global exitGlobalCount
    global timeToExecute

    fail_queries = 0

    # if reasoner_name == "unity":
    #     f = open("unification_nodes_traversed.csv", "w", newline="")
    # elif reasoner_name == "autoencoder":
    #     f = open("autoencoder_nodes_traversed.csv", "w", newline="")
    # elif reasoner_name == "chainbased":
    #     f = open("chainbased_nodes_traversed.csv", "w", newline="")
    # elif reasoner_name == "termwalk":
    #     f = open("termwalk_nodes_traversed.csv", "w", newline="")
    # elif reasoner_name == "standard":
    #     f = open("standard_nodes_traversed.csv", "w", newline="")
    # else:
    #     raise ValueError
    # writer = csv.writer(f)
    # headerColumns = ["Query", "Nodes Traversed"]
    # writer.writerow(headerColumns)
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
        # with open(log_file, 'a') as f:
        #     f.write("Query " + str(i+1) + ": " + str(query) + "\n")
        # i += 1
#        path_guide = knowledgebase.Path(query, None, None, 0)
 #       max_depth_guide = reasoner.MaxDepth(10)

        # depths = []
        # Every iteration of our backward chaining reasoner begins with executing the backwardchainguided method.
        success, guided_answers, final_path = mr_reasoner.query([query])
        if not success:
            fail_queries = fail_queries + 1

        time_end = process_time()
        timeToExecute += time_end - time_start
        exitGlobalCount = 0
        globalCount = 0

        total_nodes = mr_reasoner.num_nodes
        # row = [str(query), str(total_nodes)]
        # writer.writerow(row)

        # if depths != []:
        #     min_dep = min(depths)
        # else:
        #     min_dep = 0

        if final_path:
            sol_dep = final_path.depth
        else:
            sol_dep = 0

        guided_count_total.append(sol_dep)

        t = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
        print(
            f"{total_nodes} :: {sol_dep} - {t} ({int(total_nodes/(time_end - time_start)) if (time_end - time_start) > 0 else '-'} nps)\n")
        # open(log_file, 'a').write(
        #     f"{max_depth_guide.num_nodes} :: {min_dep} - {t} ({int(max_depth_guide.num_nodes/(time_end - time_start)) if (time_end - time_start) > 0 else '-'} nps)\n\n")

        # if reasoner_name == "unity":
        #     data_dict.append(
        #         {
        #             "query": i,
        #             "unity reasoner": reasoner_name,
        #             "unity nodes explored": nodes,
        #             "unity min depth": min_dep,
        #             "success": success,
        #             "time": time_end - time_start,
        #         }
        #     )
        # elif reasoner_name == "autoencoder":
        #     data_dict.append(
        #         {
        #             "query": i,
        #             "auto reasoner": reasoner_name,
        #             "auto nodes explored": nodes,
        #             "auto min depth": min_dep,
        #             "success": success,
        #         }
        #     )
        # elif reasoner_name == "chainbased":
        #     data_dict.append(
        #         {
        #             "query": i,
        #             "chainbased reasoner": reasoner_name,
        #             "chainbased nodes explored": nodes,
        #             "chainbased min depth": min_dep,
        #             "success": success,
        #         }
        #     )
        # elif reasoner_name == "termwalk":
        #     data_dict.append(
        #         {
        #             "query": i,
        #             "termwalk reasoner": reasoner_name,
        #             "termwalk nodes explored": nodes,
        #             "termwalk min depth": min_dep,
        #             "success": success,
        #         }
        #     )
        # else:
        #     raise ValueError
        # print()

    # f.close()
    guide_mean_total = sum(guided_count_total) / len(guided_count_total)
    print(f"{reasoner_name}: {guide_mean_total}")
    if fail_queries > 0:
        print(str(fail_queries) + " queries failed")

    data += data_dict
    return guide_mean_total



if __name__ == "__main__":

    embed_size = 50
    vocab_file = "vocab"
    # kb_file = "gameofthrones.txt"
    kb_file = "lubm-bin-benchq.txt"
    qfile = "test_queries.txt"
    # qfile = "debug_queries.txt"
    embed_model_path = "rKB_model.pth"
    guidance_model_path = "uni_mr_model.pt"

    vocab = Vocabulary()
    vocab.init_from_vocab(vocab_file)

    input_size = len(vocab.predicates) + (
        (len(vocab.variables) + len(vocab.constants)) * vocab.maxArity
    )

    kb = kbparser.parse_KB_file(kb_file)
    test_queries = kbparser.parse_KB_file(qfile).rules
    queries = [query.head for query in test_queries]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device} device")
    data1, data2, data3, data4, data5 = [], [], [], [], []

    uni_embedding = UnifierEmbed(vocab, embed_size, embed_model_path)
    uni_embedding.load()

    guidance_model = nnreasoner.NeuralNet(
        nnreasoner.hidden_size1, nnreasoner.hidden_size2, nnreasoner.num_classes
    ).to(device)
    guidance_model.load_state_dict(
        torch.load(guidance_model_path, map_location=torch.device(
                device)
        )
    )

    ming_reasoner = MetaBackChainReasoner(kb, vocab, uni_embedding, guidance_model,
                                 MetaBackChainReasoner.min_goal_selector,
                                 MetaBackChainReasoner.max_rule_selector)

    allg_reasoner = MetaBackChainReasoner(kb, vocab, uni_embedding, guidance_model,
                                 MetaBackChainReasoner.all_goals_selector,
                                 MetaBackChainReasoner.max_rule_selector, do_trace=True)

    base_reasoner = reasoner.BackChainReasoner(kb, vocab, do_trace=True)

    guidedm = guided(
        queries,
        ming_reasoner,
        data2,
        "ming"
    )

    # guidedm = guided(
    #     queries,
    #     allg_reasoner,
    #     data2,
    #     "allg"
    # )

    # base_results = guided(
    #     queries,
    #     base_reasoner,
    #     data2,
    #     "base"
    # )

    # print("Time took to execute program: " + str(timeToExecute))

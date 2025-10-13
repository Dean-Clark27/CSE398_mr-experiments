import argparse
import csv
import os
import sys
import time
from time import process_time
from types import FunctionType

from embedmodel import UnifierEmbed, TermWalkEmbed, ChainBasedEmbed

# see reasoner.py for constants used during reasoning, such as NODE_MAX

# need this to fix interrupt issues, must be before any scipy is imported
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import kbparser
from basictypes import Atom
from vocab import Vocabulary
import nnreasoner
import nnunifier
import reasoner
from reasoner import BackChainReasoner
from mr_back_reasoner import MetaBackChainReasoner
from kbparser import parse_KB_file

import torch


def eval_config(
        queries: list[list[Atom]],
        a_reasoner: BackChainReasoner,
        data: list,
        config_name="unity",
):
    # global globalCount
    # global exitGlobalCount
    # global timeToExecute

    fail_queries = 0
    total_exec_time = 0

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
    node_count_total = []

    # log_file = 'trace_log.txt' if use_alt else 'trace_log-old.txt'

    for i in range(len(queries)):
        time_start = process_time()
        query = queries[i]
        print("Query " + str(i + 1) + ": " + str(query))
        # with open(log_file, 'a') as f:
        #     f.write("Query " + str(i+1) + ": " + str(query) + "\n")
        # i += 1
        #        path_guide = knowledgebase.Path(query, None, None, 0)
        #       max_depth_guide = reasoner.MaxDepth(10)

        # depths = []
        # Every iteration of our backward chaining reasoner begins with executing the backwardchainguided method.
        success, answers, final_path = a_reasoner.query(query)
        if not success:
            fail_queries = fail_queries + 1

        time_end = process_time()
        total_exec_time += time_end - time_start
        # exitGlobalCount = 0
        # globalCount = 0

        total_nodes = a_reasoner.num_nodes
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

        node_count_total.append(total_nodes)

        t = time.strftime("%H:%M:%S", time.gmtime(time_end - time_start))
        print(
            f"{total_nodes} :: {sol_dep} - {t} ({int(total_nodes / (time_end - time_start)) if (time_end - time_start) > 0 else '-'} nps)\n")
        # open(log_file, 'a').write(
        #     f"{max_depth_guide.num_nodes} :: {min_dep} - {t} ({int(max_depth_guide.num_nodes/(time_end - time_start)) if (time_end - time_start) > 0 else '-'} nps)\n\n")

        data_dict.append(
            {
                "query": i + 1,
                config_name + " reasoner" : config_name,
                config_name + " nodes explored" : total_nodes,
                config_name + " min depth" : sol_dep,
                "success" : success,
                "time": time_end - time_start,
            }
        )
        # print()

    # f.close()
    avg_nodes = sum(node_count_total) / len(node_count_total)
    print(f"{config_name}: {avg_nodes} nodes/query")
    if fail_queries > 0:
        print(str(fail_queries) + " queries failed")
    print("Time to run all queries: " + str(total_exec_time))

    data += data_dict

    # fn = ("unity_data-i" if not args.no_goal_pruning else "unity_data") + \
    #     f"-{len(vocab.predicates)}-{len(vocab.constants)}-{vocab.maxArity}-{embed_size}.csv"
    fn = f"{config_name}-{len(vocab.predicates)}-{len(vocab.constants)}-{vocab.maxArity}-{embed_size}.csv"
    with open(fn, mode="w", newline="") as file:
        fieldnames = [
            "query",
            config_name + " reasoner",
            config_name + " nodes explored",
            config_name + " min depth",
            "success",
            "time",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        if file.tell() == 0:
            writer.writeheader()
        for row in data_dict:
            writer.writerow(row)

    return avg_nodes


def load_guidance(guidance_model_path) -> nnreasoner.NeuralNet:
    model = nnreasoner.NeuralNet(
        nnreasoner.hidden_size1, nnreasoner.hidden_size2, nnreasoner.num_classes
    ).to(device)
    model.load_state_dict(
        torch.load(
            guidance_model_path, map_location=torch.device(
                device)
        )
    )
    return model


if __name__ == "__main__":
    aparser = argparse.ArgumentParser(description="Run the experiment.")
    aparser.add_argument(
        "--kb", default="randomKB.txt", help="Name of file containing knowledge base"
    )
    aparser.add_argument(
        "--qfile", default="test_queries.txt", help="Name of file containing queries"
    )
    # aparser.add_argument(
    #     "--load_vocab",
    #     action="store_true",
    #     help="Load vocab from file instead of generating it from the KB",
    # )
    aparser.add_argument(
        "--vocab_file",
        default="vocab",
        help="Path to load initial vocabulary from. If not specified, a vocabulary will be generated from the KB.",
    )
    aparser.add_argument("-e", "--embed_size", type=int, default=50,
                         help="Embed size. Defaults to 50")

    aparser.add_argument("--search_depth", type=int, default=15,
                         help="The depth the reasoner will search to before giving up and trying another path.")

    aparser.add_argument("-s", "--standard", action="store_true")

    aparser.add_argument("-u", "--unifier", action="store_true")
    aparser.add_argument(
        "--embed_model_path",
        default="rKB_model.pth",
        help="The path to the unification embeddings model. By default, rKB_model.pth.",
    )

    aparser.add_argument(
        "--scoring_model_path",
        default="uni_mr_model.pt",
        help="The path to the guided reasoner model. By default, uni_mr_model.pth.",
    )

    aparser.add_argument("--hash_const", action="store_true",
                         help="Use hash-based constant encoding instead of one-hot.")

    aparser.add_argument("-a", "--autoencoder", action="store_true")
    aparser.add_argument(
        "--auto_model_path",
        default="auto_encoder.pth",
        help="The path to the autoencoder embeddings model. By default, auto_encoder.pth.",
    )
    aparser.add_argument(
        "--auto_guidance_model_path",
        default="auto_mr_model.pt",
        help="The path to the guided reasoner model trained using autoencoder embeddings. By default, auto_rule_classifier.pth.",
    )

    aparser.add_argument("-t", "--termwalk", action="store_true")
    # aparser.add_argument("--termwalk_guidance_model_path", default="termwalk_rule_classifier.pth", help="The path to the guided reasoner model trained using termwalk embeddings. By default, termwalk_rule_classifier.pth.")
    aparser.add_argument(
        "--termwalk_guidance_model_path",
        default="tw_mr_model.pt",
        help="The path to the guided reasoner model trained using termwalk embeddings. By default, termwalk_rule_classifier.pth.",
    )

    aparser.add_argument("-c", "--chainbased", action="store_true")
    # aparser.add_argument("--chainbased_guidance_model_path", default="chainbased_rule_classifier.pth", help="The path to the guided reasoner model trained using chainbased embeddings. By default, chainbased_rule_classifier.pth.")
    aparser.add_argument(
        "--chainbased_guidance_model_path",
        default="cb_mr_model.pt",
        help="The path to the guided reasoner model trained using chainbased embeddings. By default, chainbased_rule_classifier.pth.",
    )

    # control options
    # Notes as of 6/24/2024
    # Not using min_score is the new default
    # aparser.add_argument(
    #     "--alt_select",
    #     action="store_true",
    #     help="Use alternative control strategy of getting best goal from worst rule.",
    # )
    aparser.add_argument(
        "--config_name",
        default="default",
        help="Short name for selected experiment configuration"
    )
    aparser.add_argument(
        "--use_min_score",
        action="store_true",
        help="Indicates that all matching rules will be attempted",
    )
    aparser.add_argument(
        "--control",
        choices=["allgoals", "mingoal", "maxgoal"],
        default="mingoal",
        help="Type of control mechanism for choosing next goal",
    )
    # aparser.add_argument(
    #     "--no_goal_pruning",
    #     action="store_true",
    #     help="Indicates that goal pruning will not be used",
    # )

    aparser.add_argument(
        "--trace", action="store_true", help="Output a trace of each query"
    )
    aparser.add_argument(
        "--explain", action="store_true", help="Show the solution to each query"
    )

    args = aparser.parse_args()
    embed_size = args.embed_size  # get_embed_size(vocab)

    if args.unifier:
        if not (
                os.path.isfile(args.embed_model_path)
                and os.path.isfile(args.scoring_model_path)
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

    vocab_file = args.vocab_file + ".pkl"
    if not os.path.isfile(vocab_file):
        print("Vocab file " + vocab_file + " does not exist")
        sys.exit(1)

    vocab = Vocabulary()
    vocab.init_from_vocab(args.vocab_file)
    vocab.set_use_hash_const(args.hash_const)

    # input_size = vocab.get_input_encode_size()

    if args.kb:
        if not os.path.isfile(args.kb):
            print("Invalid KB file " + os.path.abspath(args.kb))
            sys.exit(1)
    if args.qfile:
        if not os.path.isfile(args.qfile):
            print("Invalid query file " + os.path.abspath(args.qfile))
            sys.exit(1)

    kb = kbparser.parse_KB_file(args.kb)

    queries = kbparser.parse_query_file(args.qfile)
    #queries = [query.head for query in test_queries]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device} device")
    data1, data2, data3, data4, data5 = [], [], [], [], []

    goal_selector:FunctionType
    rule_selector:FunctionType

    config_name = args.config_name

    if args.control == "mingoal":
        goal_selector = MetaBackChainReasoner.min_goal_selector
        if config_name == "default":
            config_name = "ming"
    elif args.control == "allgoals":
        goal_selector = MetaBackChainReasoner.all_goals_selector
        if config_name == "default":
            config_name = "allg"
    else:
        print(args.control + " control option not yet supported!")

    use_min_score = args.use_min_score
    if use_min_score:
        # print("Using a minimum score for rules")
        rule_selector = MetaBackChainReasoner.max_rule_with_min_scoring_selector
        config_name = config_name + "-min"
    else:
        rule_selector = MetaBackChainReasoner.max_rule_selector

    # run the selected models on the test queries: standard, unifier, autoencoder, termwalk, and/or chain-based

    if args.standard:
        print("STANDARD\n")
        base_config = BackChainReasoner(kb, vocab, max_depth=args.search_depth,
                                        do_trace=args.trace, print_solution=args.explain)
        base_results = eval_config(queries, base_config, data1, "std")
        # base(queries, "standard", data1, args.kb, args.trace)

    # assumes that the appropriate embeddings and meta reasoning models have been trained already
    if args.unifier:
        print("UNITY: " + config_name)
        print("\tEmbedding Model: " + args.embed_model_path)
        print("\tScoring Model: " + args.scoring_model_path)
        print()
        print("\tControl: " + args.control)
        if args.control == "allgoals":
            print("\t\tNote: Typically should be used with a scoring model that wasn't training on negative facts")
        if use_min_score:
            print("\tRule Eval: Max Rule with Minimum Scoring")
        else:
            print("\tRule Eval: Max Rule")

        uni_embedding = UnifierEmbed(vocab, embed_size, args.embed_model_path)
        uni_embedding.load()

        # load the guidance model
        guidance_model = load_guidance(args.scoring_model_path)

        #TODO: does the call below need an embed_size argument?
        uni_reasoner = MetaBackChainReasoner(kb, vocab, uni_embedding, guidance_model,
                                             goal_selector, rule_selector, max_depth=args.search_depth,
                                             do_trace=args.trace, print_solution=args.explain)

        uni_results = eval_config(
            queries,
            uni_reasoner,
            data2,
            config_name
        )


    if args.autoencoder:
        print("Autoencoder temporarily disabled.")
        # model_path = args.auto_model_path
        # print("AUTO")
        # print("\tEmbedding Model: " + model_path)
        # print("\tReasoning Model: " + args.auto_guidance_model_path)
        # whole_model = autoencoder.NeuralNet().to(device)
        # whole_model.load_state_dict(
        #     torch.load(model_path, map_location=torch.device(device))
        # )
        # whole_model.eval()
        # embed_model = whole_model.encoder
        # guidance_model = nnreasoner.NeuralNet(
        #     nnreasoner.hidden_size1, nnreasoner.hidden_size2, nnreasoner.num_classes
        # ).to(device)
        # guidance_model.load_state_dict(
        #     torch.load(args.auto_guidance_model_path,
        #                map_location=torch.device(device))
        # )
        # guidedm = guided(
        #     queries,
        #     embed_model,
        #     guidance_model,
        #     "autoencoder",
        #     data3,
        #     args.kb,
        #     args.alt_select,
        #     args.trace,
        #     use_min_score,
        # )
        #

    if args.termwalk:
        print("TERMWALK")
        print("\tScoring Model: " + args.termwalk_guidance_model_path)
        print()
        print("\tControl: " + args.control)
        if args.control == "allgoals":
            print("\t\tNote: Typically should be used with a scoring model that wasn't training on negative facts")
        if use_min_score:
            print("\tRule Eval: Max Rule with Minimum Scoring")
        else:
            print("\tRule Eval: Max Rule")

        tw_embedding = TermWalkEmbed(vocab)

        # load the guidance model
        guidance_model = load_guidance(args.termwalk_guidance_model_path)

        tw_reasoner = MetaBackChainReasoner(kb, vocab, tw_embedding, guidance_model,
                                            goal_selector, rule_selector,
                                            do_trace=args.trace, print_solution=args.explain)

        config_name = "tw-" + config_name
        tw_results = eval_config(
            queries,
            tw_reasoner,
            data4,
            config_name
        )

    if args.chainbased:
        print("CHAINBASED")
        print("\tScoring Model: " + args.chainbased_guidance_model_path)
        print()
        print("\tControl: " + args.control)
        if args.control == "allgoals":
            print("\t\tNote: Typically should be used with a scoring model that wasn't training on negative facts")
        if use_min_score:
            print("\tRule Eval: Max Rule with Minimum Scoring")
        else:
            print("\tRule Eval: Max Rule")

        cb_embedding = ChainBasedEmbed(vocab, embed_size)

        # load the guidance model
        guidance_model = load_guidance(args.chainbased_guidance_model_path)

        cb_reasoner = MetaBackChainReasoner(kb, vocab, cb_embedding, guidance_model,
                                            goal_selector, rule_selector,
                                            do_trace=args.trace, print_solution=args.explain)

        config_name = "cb-" + config_name
        cb_results = eval_config(
            queries,
            cb_reasoner,
            data5,
            config_name
        )

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


import knowledgebase
import random
import basictypes
import atomgenerator
import kbparser
import chainbased
import argparse
import torch
import numpy as np
import nnunifier
import reasoningenv
import dqn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--generate_queries", action="store_true",
                        help="Generate new queries instead of reading them \
                        in.")
    parser.add_argument("-n", "--num_queries", type=int, default=200,
                        help="Number of queries. Default: 200")
    parser.add_argument("--train_query_path", default="train_queries.txt",
                        help="Path to save the list of queries used for \
                        training. Default: train_queries.txt")
    parser.add_argument("--test_query_path", default="test_queries.txt",
                        help="Path to save the list of queries used to test \
                        the model. Default: test_queries.txt")
    parser.add_argument("-t", "--train", action="store_true",
                        help="Train the model.")
    parser.add_argument("--embedding", choices=["chainbased", "unification"],
                        default="chainbased", help="Default: chainbased")
    parser.add_argument("--max_depth", type=int, default=4,
                        help="Default: 4.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Default: 1e-4")
    parser.add_argument("-i", "--iterations", type=int, default=20,
                        help="Number of times to train per query. Default: 20")
    parser.add_argument("--epsilon_start", type=float, default=0.9,
                        help="Default: 0.9")
    parser.add_argument("--epsilon_end", type=float, default=0.05,
                        help="Default: 0.05")
    parser.add_argument("--epsilon_decay", type=int, default=2000,
                        help="Default: 2000")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Default: 128")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Default: 0.99")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Default: 0.005")
    parser.add_argument("-s", "--save_model", default="rl-policy.pth",
                        help="Path to save the model.")
    parser.add_argument("-e", "--test", action="store_true",
                        help="Test the trained or loaded model on the test \
                        queries.")
    parser.add_argument("--alternative_select", action="store_true",
                        help="Use the alternative method of selecting actions.")
    parser.add_argument("--omit_standard", action="store_true",
                        help="Do not run the experiment using the standard \
                        reasoner.")
    parser.add_argument("--no_backtrack", action="store_true",
                        help="Do not backtrack. Use if in a hurry.")
    parser.add_argument("--omit_guided", action="store_true",
                        help="Do not run the experiment using the guided \
                        reasoner.")
    parser.add_argument("-l", "--load_model",
                        help="Path to load the model.")
    parser.add_argument("--unification_model", default="rKB_model.pth",
                        help="Path to the unification embedding model. \
                        Default: rKB_model.pth")
    parser.add_argument("--csv_name", default="rl",
                        help="Format for the csv files. Will append postfix.\
                        Default: rl")
    parser.add_argument("-k", "--knowledge_base", default="randomKB.txt",
                        help="Path to the knowledge base. \
                        Default: randomKB.txt")
    parser.add_argument("-f", "--facts_list", default="random_facts.txt",
                        help="Path to the list of facts. \
                        Default: random_facts.txt")
    parser.add_argument("-v", "--verbose", type=int, default=1,
                        help="Print out what the reasoner is doing. 0 prints \
                        out nothing; 1 prints out what the program is doing; \
                        2 prints out debug info from the environment. \
                        Default: 1")
    args = parser.parse_args()

    global policy_net
    random.seed(0)

    if args.train or args.test:
        # Define embedding functions
        def chainbased_embed(rule):
            return chainbased.represent_pattern(rule, 10)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        unification_model = nnunifier.NeuralNet(nnunifier.input_size,
                                                nnunifier.hidden_size1,
                                                nnunifier.hidden_size2,
                                                nnunifier.num_classes).to(device)
        unification_model.load_state_dict(torch.load(args.unification_model,
                                                     map_location=torch.device(device)))
        unification_model.eval()

        def unification_embed(rule):
            if isinstance(rule, knowledgebase.Rule):
                if rule.head.predicate.name == "yes":
                    with torch.no_grad():
                        embedding = torch.zeros(20).to(device)
                        for atom in rule.body:
                            embedding = torch.add(embedding, unification_model(
                                torch.FloatTensor(atomgenerator.encodeAtom(atom)).to(device))) #Spencer: encodeAtom
                        embedding = embedding.cpu().detach().numpy()
                else:
                    with torch.no_grad():
                        head = unification_model(torch.FloatTensor(
                            atomgenerator.encodeAtom(rule.head)).to(device)).cpu().detach().numpy() #Spencer: encodeAtom
                        body = torch.zeros(20).to(device)
                        for atom in rule.body:
                            body = torch.add(body, unification_model(torch.FloatTensor(
                                atomgenerator.encodeAtom(atom)).to(device))) #Spencer: encodeAtom
                        body = body.cpu().detach().numpy()
                        embedding = np.concatenate([head, body])
                return embedding
            elif isinstance(rule, basictypes.Atom):
                with torch.no_grad():
                    embedding = unification_model(torch.FloatTensor(
                        atomgenerator.encodeAtom(rule)).to(device)).cpu().detach().numpy() #Spencer: encodeAtom
                    return embedding
            else:
                raise TypeError

        if args.embedding == "chainbased":
            embed = chainbased_embed
        elif args.embedding == "unification":
            embed = unification_embed
        else:
            raise ValueError

    if args.train:
        if args.verbose >= 1:
            print("Training model")

        # Read in the knowledge base
        kb = kbparser.parse_KB_file(args.knowledge_base)
        if not args.generate_queries:
            train_queries = kbparser.parse_KB_file(args.train_query_path)

        # Get the input size of the model
        input_size = dqn.get_input_size(kb, train_queries, embed)

        # Set up the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy_net = dqn.DQN(input_size).to(device)

        dqn.train_model(policy_net, kb, train_queries, embed, device, args.max_depth,
                        args.learning_rate, args.iterations,
                        args.epsilon_start, args.epsilon_end,
                        args.epsilon_decay, args.batch_size, args.gamma,
                        args.tau, args.verbose)
        if args.verbose >= 1:
            print(f"Saving model to {args.save_model}")
        torch.save(policy_net.state_dict(), args.save_model)

    if args.test:
        if args.verbose >= 1:
            print("Running experiment")
        if not args.train:
            kb = kbparser.parse_KB_file(args.knowledge_base)

        if not args.generate_queries:
            test_queries = kbparser.parse_KB_file(args.test_query_path)

        if not args.train and not args.omit_guided:
            input_size = dqn.get_input_size(kb, test_queries, embed)

            # Set up the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            policy_net = dqn.DQN(input_size).to(device)
            policy_net.load_state_dict(torch.load(args.load_model))

        if not args.omit_guided:
            policy_net.eval()

        if not args.omit_standard:
            if args.verbose >= 1:
                print("Standard")
            reasoningenv.standardreasoner(kb, test_queries, args.max_depth, args.verbose,
                                          f"{args.csv_name}_standard.csv")

        if not args.omit_guided:
            if args.verbose >= 1:
                print("Guided")
            if not args.no_backtrack:
                if not args.alternative_select:
                    dqn.guidedreasoner(kb, test_queries, policy_net, embed,
                                       dqn.select_action_dqn, args.max_depth, args.verbose,
                                       f"{args.csv_name}_{args.embedding}_guided.csv")
                else:
                    dqn.guidedreasoner(kb, test_queries, policy_net, embed,
                                       dqn.select_action_dqn_lowest, args.max_depth, args.verbose,
                                       f"{args.csv_name}_{args.embedding}_altguided.csv")
            else:
                dqn.guidedreasoner_nobacktrack(kb, test_queries, policy_net,
                                               embed, dqn.select_action_dqn,
                                               args.verbose, f"{args.csv_name}_{args.embedding}_nobacktrack_guided.csv")

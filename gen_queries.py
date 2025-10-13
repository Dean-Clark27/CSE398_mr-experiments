from copy import copy
from random import shuffle
from basictypes import Constant, Variable
from kbparser import parse_KB_file
from vocab import Vocabulary

vocab = Vocabulary()


def generate_variable_combinations(input_file):
    # Read the input file
    facts = parse_KB_file(input_file)
    vocab.init_from_vocab()
    queries: list[str] = []

    for a in facts.rules:
        cycle = False
        for arg in a.head.arguments:
            # JDH: what is the purpose of this line? how can a constant not be in the set of known constants?
            if isinstance(arg, Constant) and arg not in vocab.constants:
                cycle = True
                break
        if cycle:
            continue

        if a.head.arity == 1:
            # f.write(str(a.head) + ".\n")
            queries.append(str(a.head) + ".\n")
            a.head.arguments[0] = Variable('X')
            # f.write(str(a.head) + ".\n")
            queries.append(str(a.head) + ".\n")
        elif a.head.arity == 2:
            # f.write(str(a.head) + ".\n")
            queries.append(str(a.head) + ".\n")
            args = copy(a.head.arguments)
            a.head.arguments[0] = Variable('X')
            # f.write(str(a.head) + ".\n")
            queries.append(str(a.head) + ".\n")
            a.head.arguments[0] = args[0]
            a.head.arguments[1] = Variable('Y')
            # f.write(str(a.head) + ".\n")
            queries.append(str(a.head) + ".\n")
            a.head.arguments[0] = Variable('X')
            a.head.arguments[1] = Variable('Y')
            # f.write(str(a.head) + ".\n")
            queries.append(str(a.head) + ".\n")
    shuffle(queries)
    i = 0
    # TODO: what if we have fewer than 200 lines in the source file?
    # with open("train_queries.txt", 'w') as f:
    #     while i < 100:
    #         f.write(queries[i])
    #         i += 1
    with open("test_queries.txt", 'w') as f:
        while i < 100:
            f.write(queries[i])
            i += 1


if __name__ == "__main__":
    # generate_variable_combinations('../../Data/lubm_query_from_q14.txt')
    generate_variable_combinations('lubm_query_from_q14.txt')
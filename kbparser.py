from basictypes import Constant, Variable, Predicate, Atom
from knowledgebase import KnowledgeBase, Rule
import os
import sys
import re

# Prolog syntax
# consants start lower case
# variables start upper case
# period at the end optional, but not reccomended


def parse_arguments(arg_expr) -> list:
    if (len(arg_expr) == 1):
        return []
    arg_list = re.split(",", arg_expr[:len(arg_expr)-1])

    for i in range(len(arg_list)):
        arg_list[i] = arg_list[i]

        if arg_list[i][0].islower():
            arg_list[i] = Constant(arg_list[i])  # Add constant to parser list
        else:
            arg_list[i] = Variable(arg_list[i])  # Add variable to parser list
    return arg_list


def parse_atom(atom_expr) -> Atom:
    atom_expr = "".join(atom_expr.split())       # this removes all whitespace
    pred_args = atom_expr.split("(")
    if len(pred_args) < 2:
        print("Error parsing atom: " + atom_expr)
    pred = pred_args[0]

    args = pred_args[1]
    args = parse_arguments(args)
    return Atom(Predicate(len(args), pred), args)


def parse_rule(rule_expr) -> Rule:
    rule_list = rule_expr.split(":-")
    head = rule_list[0]
    head = parse_atom(head)
    if (len(rule_list) == 1):
        return Rule(head, [])
    body_expr = re.split("(?<=\)),", rule_list[1])

    body = []
    for atom in body_expr:
        body.append(parse_atom(atom))
    return Rule(head, body)

# reads a file and adds the rules to a knowledge base

def parse_query(query_expr):
    query_atoms = re.split("(?<=\)),", query_expr)
    query = []
    for atom in query_atoms:
        query.append(parse_atom(atom)) 
    return query

def write_queries(queries: list[list[Atom]], path):
    with open(path, "w") as f:
        for query in queries:
            querystr = ""
            for i in range(len(query)):
                querystr += str(query[i])
                if i < len(query)-1:
                    querystr += ", "
            f.write(querystr+"."+"\n")


def parse_KB_file(file_path):
    KB = KnowledgeBase([])

    with open(file_path, mode='r') as f:

        lines = f.readlines()
        comments = []
        lines = map(lambda x: x.lstrip(), lines)
        lines = [x for x in lines if not (len(x) == 0 or x[0] == "%")]

        lines = "".join(lines)

        # TODO: Catch parse errors where the "." is missing
        lines = lines.split(".")
    for line in lines:
        line = ''.join(line.split())        # remove internal whitespace
        line = line.strip()
        if len(line) == 0:
            continue
        rule = parse_rule(line)

        KB.addrule(rule)

    return KB

# writes KB to a file

def parse_query_file(file_path):
    collq = []
    with open(file_path, mode='r') as f:
        lines = f.readlines()
        comments = []
        lines = map(lambda x: x.lstrip(), lines)
        lines = [x for x in lines if not (len(x) == 0 or x[0] == "%")]

        lines = "".join(lines)

        lines = lines.split(".")
    for line in lines:
        line = ''.join(line.split())        # remove internal whitespace
        line = line.strip()
        if len(line) == 0:
            continue
        result = parse_query(line)
        collq.append(result)
    return collq


def KB_to_txt(KB: KnowledgeBase, path):
    with open(path, "w") as f:
        for rule in KB.rules:
            f.write(str(rule)+"."+"\n")


# parses and manipulates knowledge bases
"""if __name__ == "__main__":
    kb = parse_KB_file(os.path.join(sys.path[0], "gameofthrones.txt"))
    #kb.print()
    KB_to_txt(kb, "test.txt")
    testFile = parse_KB_file(os.path.join(sys.path[0], "test.txt"))
    testFile.print()
"""

if __name__ == "__main__":
    pass
    # kb = thisParser.parse_KB_file(os.path.join(sys.path[0],"gameofthrones.txt"))
    # kb.print()
    # thisParser.KB_to_txt(kb, "test.txt")

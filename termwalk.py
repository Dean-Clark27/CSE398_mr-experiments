# File: termwalk.py
# Description: functions and classes to help represent
# Horn logic clauses through the 3-term-walk representation
# described in Jakubuv and Urban's 2017 paper (ENIGMA)
import networkx as nx
from numpy import ndarray

import knowledgebase
from basictypes import Constant, Predicate, Variable, Atom
import numpy as np
from copy import deepcopy

from vocab import Vocabulary


# Should be used to represent affirmation,
# negation,  logical disjunction,
# or logical conjunction.
class Symbol:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


def graph_from_rule(rule: knowledgebase.Rule):
    """Given an input rule, returns the corresponding digraph of its
    symbols. For example, the rule p0(a, b) :- p2(x) would have nodes
    with the symbols +, logical disjunction, +, p0, a, b, -, p2, and x."""
    pos = Symbol("+")
    neg = Symbol("-")
    disj = Symbol("or")

    graph_index = 0
    root_index = graph_index
    G = nx.DiGraph()

    def graph_from_atom(atom: Atom):
        nonlocal graph_index
        nonlocal G
        predicate_index = graph_index
        G.add_node(graph_index)
        G.nodes[graph_index]["symbol"] = deepcopy(atom.predicate)
        graph_index += 1

        for term in atom.arguments:
            G.add_node(graph_index)
            G.nodes[graph_index]["symbol"] = deepcopy(term)
            G.add_edge(predicate_index, graph_index)
            graph_index += 1

        return predicate_index

    def pos_graph_from_atom(atom: Atom):
        nonlocal graph_index
        nonlocal G
        pos_index = graph_index
        G.add_node(graph_index)
        G.nodes[graph_index]["symbol"] = pos
        graph_index += 1

        next = graph_from_atom(atom)
        G.add_edge(pos_index, next)

        return pos_index

    def neg_graph_from_atom(atom: Atom):
        nonlocal graph_index
        nonlocal G
        neg_index = graph_index
        G.add_node(graph_index)
        G.nodes[graph_index]["symbol"] = neg
        graph_index += 1

        next = graph_from_atom(atom)
        G.add_edge(neg_index, next)

        return neg_index

    G.add_node(graph_index)
    G.nodes[graph_index]["symbol"] = pos
    graph_index += 1

    if not rule.body:
        next = graph_from_atom(rule.head)
        G.add_edge(root_index, next)
    else:
        # Add a node for the disjunction of
        # all of the individual atoms
        disj_index = graph_index
        G.add_node(graph_index)
        G.nodes[graph_index]["symbol"] = disj
        G.add_edge(root_index, graph_index)
        graph_index += 1

        next = pos_graph_from_atom(rule.head)
        G.add_edge(disj_index, next)

        for atom in rule.body:
            next = neg_graph_from_atom(atom)
            G.add_edge(disj_index, next)

    return G


def graph_from_atom(atom: Atom):
    """Standalone version of pos_graph_from_atom()
    function defined within graph_from_rule(). Returns
    a graph of an atom with a positive symbol at the start."""
    graph_index = 0
    G = nx.DiGraph()

    G.add_node(graph_index)
    G.nodes[graph_index]["symbol"] = Symbol("+")
    pos_index = graph_index
    graph_index += 1

    predicate_index = graph_index
    G.add_node(graph_index)
    G.nodes[graph_index]["symbol"] = deepcopy(atom.predicate)
    G.add_edge(pos_index, predicate_index)
    graph_index += 1

    for term in atom.arguments:
        G.add_node(graph_index)
        G.nodes[graph_index]["symbol"] = deepcopy(term)
        G.add_edge(predicate_index, graph_index)
        graph_index += 1

    return G


def find_all_paths(G: nx.DiGraph, n: int):
    '''Return a list of all walks of length n in G.'''
    def findPaths(G: nx.DiGraph, node, n: int):
        if n == 0:
            return []
        if n == 1:
            return [[node]]
        paths = [[node] + path for successor in G.successors(node) for path in findPaths(G, successor, n-1)]
        return paths

    all_paths = []
    for node in G:
        all_paths.extend(findPaths(G, node, 3))

    return all_paths


def return_index(x, vocab:Vocabulary) -> int:
    '''Return the index of a variable, constant, or predicate. The order
    is determine by the vocab object, which sorts each type of symbol alphabetically
    '''
    if isinstance(x, Variable):
        return vocab.variables.index(x)
    elif isinstance(x, Constant):
        return vocab.constants.index(x)
    elif isinstance(x, Predicate):
        return vocab.predicates.index(x)
    elif isinstance(x, Symbol):
        return -1
    else:
        raise TypeError


def termwalk_representation(rule, vocab:Vocabulary,
                            behavior=0) -> ndarray:
    """Assumes that the predicates, variables,
    and constants are named uniformly in the format,
    e.g., p7, X3, a76, respectively.
    The behavior argument controls whether or not
    the representation takes into account the fact
    that the first symbol will always be either
    be either + or -."""
    predicate_index_begin = 0
    predicate_count = len(vocab.predicates)
    variable_count = len(vocab.variables)
    constant_count = len(vocab.constants)
    variable_index_begin = predicate_count
    constant_index_begin = predicate_count + variable_count
    pos_index = predicate_count + variable_count + constant_count
    neg_index = pos_index + 1
    disj_index = neg_index + 1

    symbol_count = predicate_count + variable_count + constant_count + 3
    # TODO: This appears to hard-code behavior=0. Can't handle having a symbol in the first position
    r = np.zeros((3, symbol_count, symbol_count), dtype=np.float32)

    if isinstance(rule, knowledgebase.Rule):
        G = graph_from_rule(rule)
    elif isinstance(rule, Atom):
        G = graph_from_atom(rule)
    else:
        raise TypeError

    path_list = find_all_paths(G, 3)

    for path in path_list:
        x = G.nodes[path[0]]["symbol"]
        y = G.nodes[path[1]]["symbol"]
        z = G.nodes[path[2]]["symbol"]

        a = []

        if behavior == 0:
            assert isinstance(x, Symbol)
            if x.name == "+":
                a.append(0)
            elif x.name == "-":
                a.append(1)
            elif x.name == "or":
                a.append(2)
            else:
                raise ValueError
        else:
            count = return_index(x, vocab)
            if isinstance(x, Constant):
                a.append(constant_index_begin + count)
            elif isinstance(x, Variable):
                a.append(variable_index_begin + count)
            elif isinstance(x, Predicate):
                a.append(predicate_index_begin + count)
            elif isinstance(x, Symbol):
                if x.name == "+":
                    a.append(pos_index)
                elif x.name == "-":
                    a.append(neg_index)
                elif x.name == "or":
                    a.append(disj_index)
                else:
                    raise ValueError
            else:
                raise ValueError

        for symbol in [y, z]:
            count = return_index(symbol, vocab)
            if isinstance(symbol, Constant):
                a.append(constant_index_begin + count)
            elif isinstance(symbol, Variable):
                a.append(variable_index_begin + count)
            elif isinstance(symbol, Predicate):
                a.append(predicate_index_begin + count)
            elif isinstance(symbol, Symbol):
                if symbol.name == "+":
                    a.append(pos_index)
                elif symbol.name == "-":
                    a.append(neg_index)
                elif symbol.name == "or":
                    a.append(disj_index)
                else:
                    raise ValueError
            else:
                raise ValueError

        assert len(a) == 3

        r[a[0]][a[1]][a[2]] += 1

    r = r.reshape(r.size)

    return r

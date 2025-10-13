import termwalk
import basictypes
import knowledgebase
import networkx as nx
import hashlib
import numpy as np


def chainbased_graph_from_atom(atom: basictypes.Atom):
    """Returns a graph of an atom. Note that the first
    node is indexed at 1 instead of 0, since the 0 node
    containing the + symbol is removed."""
    a = termwalk.graph_from_atom(atom)
    a.remove_node(0)

    for node in a:
        if isinstance(a.nodes[node]["symbol"], basictypes.Variable):
            a.nodes[node]["symbol"].name = "*"

    return a


def get_patterns(G: nx.DiGraph):
    """Takes a chainbased graph of an atom as input
    and outputs a list of graphs with each pattern."""

    # Note: It is possible that Crouse et al. (AAAI 21), meant for a pattern to be
    # a tree and not a sequence (see Fig. 2). It includes a single path from predicate
    # to constant/variable, but each subtree not included is replaced by a "*" node
    # Top of p. 4: "Argument position is also indicated with the use of wild-card symbols."
    # If this is the case, is the tree serialized in prefix, infix, or postfix order?

    patterns_list = []

    for i in G.successors(1):
        patterns_list.append([1, i])

    return patterns_list


def pattern_string(G: nx.DiGraph, pattern: list):
    """Takes a pattern as input and outputs its string representation."""
    ps = ""
    for i in pattern:
        ps = ps + str(G.nodes[i]["symbol"])
    return ps


def rule_pattern_strings(rule: knowledgebase.Rule):
    a = [chainbased_graph_from_atom(rule.head)]
    b = []
    for atom in rule.body:
        b.append(chainbased_graph_from_atom(atom))
    head_patterns = []
    for graph in a:
        p = get_patterns(graph)
        for i in p:
            head_patterns.append(pattern_string(graph, i))
    body_patterns = []
    for graph in b:
        p = get_patterns(graph)
        for i in p:
            body_patterns.append(pattern_string(graph, i))
    return head_patterns, body_patterns


def atom_pattern_strings(atom: basictypes.Atom):
    G = chainbased_graph_from_atom(atom)
    p = get_patterns(G)
    patterns = []
    for i in p:
        patterns.append(pattern_string(G, i))
    return patterns


def represent_pattern(rule, dimensionality: int):
    """Get the chain-based representation of a rule or an atom"""

    # From Crouse et al. AAAI 2021
    # We obtain a d-dimensional representation of a clause by
    # hashing the linearization of each pattern p using MD5 hashes
    # to compute a hash value v, and setting the element at index
    # v mod d to the number of occurrences of the pattern p in
    # the clause.

    representation = np.zeros(dimensionality, dtype=np.float32)
    if isinstance(rule, knowledgebase.Rule):
        a, b = rule_pattern_strings(rule)
        for pattern in a:
            v = int(hashlib.md5(pattern.encode('utf-8')).hexdigest(), 16)
            representation[v % dimensionality] += 1
        body_representation = np.zeros(dimensionality, dtype=np.float32)
        for pattern in b:
            v = int(hashlib.md5(pattern.encode('utf-8')).hexdigest(), 16)
            body_representation[v % dimensionality] += 1
        return np.concatenate((representation, body_representation))
    elif isinstance(rule, basictypes.Atom):
        a = atom_pattern_strings(rule)
        for pattern in a:
            v = int(hashlib.md5(pattern.encode('utf-8')).hexdigest(), 16)
            representation[v % dimensionality] += 1
        return representation
    else:
        raise TypeError

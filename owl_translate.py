import re

from rdflib import *
from rdflib.namespace import RDFS, RDF, OWL
import argparse
import os
from urllib.parse import urlparse


def extract_upper_and_numbers(string):
    extracted_chars = ''.join(char for char in string if char.isnumeric() or char.isupper())
    return extracted_chars


def clean_string(s):
    """ Make a string usable as a Datalog constant. Ensure first character is a lowercase letter,
    that all spaces are replaced with underscores, and that long text is appropriately shortened.
    """
    # remove special characters and numbers
    s = re.sub(r'[^a-zA-Z\s]', '', s)
    # replace spaces with underscores
    s = s.replace(' ', '_')
    s = s.replace('.', '_')
    # abbreviate if the string is too long
    words = s.split('_')
    if len(words) > 3:
        s = '_'.join(words[:2] + [extract_upper_and_numbers(''.join(words[2:]))])
    # convert to camelCase
    if len(s) > 1:
        return s[0].lower() + s[1:]
    else:
        return s


def catToPred(name, var, asPred=True):
    """
    convert a category to a predicate or object representation

    Args:
        name (str): the name of the category
        var (str): the variable to use in the predicate
        asPred (bool): if True, return category as predicate. if False, return as object

    Returns:
        str: the category represented as either a predicate or an object
    """
    if asPred:
        return f"{name}({var})"
    else:
        return f"type({var}, {name})"


def parse_object(uri):
    """
    parse a URI to extract a simplified name

    Args:
        uri (str): the URI to parse

    Returns:
        str: the simplified name with the first character in lowercase
    """

    # TODO: figure out how to incorporate namespaces to avoid conflicts when two ontologies have the same local name
    if '#' in uri:
        name = uri.rsplit('#', 1)[-1]
    elif '/' in uri:
        name = uri.rsplit('/', 1)[-1]
    else:
        name = uri
    if len(name) > 1:
        return clean_string(name)
    else:
        return uri


def traverse_nodes(g, node):
    """
    explore and return nodes connected to BNodes

    Args:
        g (Graph): RDF Graph instance
        node (BNode): the starting blank node

    Returns:
        list: connected nodes
    """
    linked_nodes = []
    while node and node != RDF.nil:
        item = g.value(node, RDF.first)
        linked_nodes.append(item)
        node = g.value(node, RDF.rest)
    return linked_nodes


def process_expression_body(g, s, blank_node, asPred=True):
    sub_rules = []
    on_property = parse_object(str(g.value(blank_node, OWL.onProperty, None)))

    if (blank_node, RDF.type, OWL.Restriction) in g:
        for p, o in g.predicate_objects(blank_node):
            if p == OWL.someValuesFrom:
                class_of_value = parse_object(str(o))
                sub_rules.append(f"{on_property}(X, Y), {catToPred(class_of_value, 'Y', asPred)}")
            elif p == OWL.minCardinality:
                pass
    if sub_rules:
        return sub_rules
    elif (blank_node, OWL.allValuesFrom, None) in g:
        subj = parse_object(s)
        obj = parse_object(str(g.value(blank_node, OWL.someValuesFrom, None)))
        print(f"∀ Universal restriction on the body cannot be translated: all values of '{on_property}' "
              f"for '{subj}' must be '{obj}'")
    return []


def process_expression_head(g, s, blank_node, asPred=True):
    sub_rules = []
    if (blank_node, RDF.type, OWL.Restriction) in g:
        for p, o in g.predicate_objects(blank_node):
            if p == OWL.allValuesFrom:
                class_of_value = parse_object(str(o))
                sub_rules.append(catToPred(class_of_value, 'X', asPred))

    if sub_rules:
        return sub_rules
    elif (blank_node, OWL.someValuesFrom, None) in g:
        subj = parse_object(s)
        on_property = parse_object(str(g.value(blank_node, OWL.onProperty, None)))
        obj = parse_object(str(g.value(blank_node, OWL.someValuesFrom, None)))
        print(f"∃ Existential Restriction on the head cannot be translated: there exists "
              f"some '{obj}' that '{subj}' '{on_property}'")
    return []


def process_body(g, s, node, asPred=True):
    """
    process the body of a subclass axiom

    Args:
        g (Graph): RDF Graph instance
        s: subject of the triple
        node:tThe node to process
        asPred (bool): if True, use category as predicate. if False, use as object

    Returns:
        list: Processed body rules.
    """
    if isinstance(node, BNode):
        return process_expression_body(g, s, node, asPred)
    else:
        return [catToPred(parse_object(node), 'X', asPred)]


def process_head(g, s, node, is_range, asPred=True):
    """
    process the head of a subclass axiom

    Args:
        g (Graph): RDF Graph instance
        s: subject of the triple
        node: the node to process
        is_range (bool): whether the predicate is range
        asPred (bool): if True, use category as predicate. if False, use as object

    Returns:
        str: Processed head rule.
    """
    if isinstance(node, BNode):
        return process_expression_head(g, s, node, asPred)
    elif is_range:
        return catToPred(parse_object(node), 'Y', asPred)
    else:
        return catToPred(parse_object(node), 'X', asPred)


def process_intersection(g, s, o, rules, asPred=True):
    """
    process an intersection of classes

    Args:
        g (Graph): RDF Graph instance
        s: subject of the triple
        o: object of the triple
        rules (list): list to append generated rules
        asPred (bool): if True, use category as predicate. if False, use as object
    """
    subj = parse_object(s)
    blank_nodes = traverse_nodes(g, o)
    sub_rules = []
    for node in blank_nodes:
        if isinstance(node, BNode):
            sub_rules.extend(process_body(g, s, node, asPred))
        else:
            class_name = parse_object(node)
            sub_rules.append(catToPred(class_name, 'X', asPred))
            rules.append(f"{catToPred(class_name, 'X', asPred)} :- {catToPred(subj, 'X', asPred)}.")
    if sub_rules:
        rules.append(f"{catToPred(subj, 'X', asPred)} :- {', '.join(sub_rules)}.")


def process_subclass(g, s, o, rules, asPred=True):
    """
    process a subclass axiom

    Args:
        g (Graph): RDF Graph instance
        s: subject of the triple
        o: object of the triple
        rules (list): list to append generated rules
        asPred (bool): if True, use category as predicate. if False, use as object
    """
    subj_rules = process_body(g, o, s, asPred)
    obj_rules = process_head(g, s, o, False, asPred)

    if obj_rules:
        obj_rule_str = ', '.join(obj_rules) if isinstance(obj_rules, list) else obj_rules
        subj_rule_str = ', '.join(subj_rules) if isinstance(subj_rules, list) else subj_rules
        rules.append(f"{obj_rule_str} :- {subj_rule_str}.")


def process_triple(g, s, p, o, rules, asPred=True):
    """
    Process a single triple in the ontology graph.

    Args:
        g (Graph): RDF Graph instance
        s: subject of the triple
        p: predicate of the triple
        o: object of the triple
        rules (list): list to append generated rule
        asPred (bool): if True, use category as predicate. igf False, use as object
    """
    ignore = {'first', 'rest', 'nil', 'label', 'objectProperty', 'datatypeProperty', 'class', 'versionInfo',
              'comment', 'ontology', 'intersectionOf', 'someValuesFrom', 'allValuesFrom', 'restriction', 'onProperty'}

    if p == OWL.intersectionOf:
        process_intersection(g, s, o, rules, asPred)

    elif p == RDFS.subClassOf:
        process_subclass(g, s, o, rules, asPred)

    elif p == OWL.equivalentClass:
        if isinstance(s, BNode) or isinstance(o, BNode):
            print(f"equivalentClass axiom involving complex expressions cannot be translated: {s} ≡ {o}")
        else:
            process_subclass(g, s, o, rules, asPred)
            process_subclass(g, o, s, rules, asPred)

    elif p == RDFS.subPropertyOf:
        subj = parse_object(s)
        obj = parse_object(o)
        rules.append(f"{obj}(X, Y) :- {subj}(X, Y).")

    elif p == RDFS.domain:
        subj = parse_object(s)
        obj_rules = process_head(g, s, o, False, asPred)
        if obj_rules:
            obj_rule_str = ', '.join(obj_rules) if isinstance(obj_rules, list) else obj_rules
            rules.append(f"{obj_rule_str} :- {subj}(X, Y).")
        else:
            print("Class expressions not supported in head of range")

    elif p == RDFS.range:
        subj = parse_object(s)
        obj_rules = process_head(g, s, o, True, asPred)
        if obj_rules:
            obj_rule_str = ', '.join(obj_rules) if isinstance(obj_rules, list) else obj_rules
            rules.append(f"{obj_rule_str} :- {subj}(X, Y).")
        else:
            print("Class expressions not supported in head of range")

    elif p == RDF.type and o == OWL.TransitiveProperty:
        subj = parse_object(s)
        rules.append(f"{subj}(X, Z) :- {subj}(X, Y), {subj}(Y, Z).")

    elif p == OWL.inverseOf:
        subj = parse_object(s)
        obj = parse_object(o)
        rules.append(f"{obj}(X, Y) :- {subj}(Y, X).")
        rules.append(f"{subj}(X, Y) :- {obj}(Y, X).")

    elif not any(parse_object(term) in ignore for term in [p, o]):
        pred = parse_object(p)
        obj = parse_object(o)
        print(f"{s}, {p}, {o}: '{pred}' '{obj}' is not accepted")


def main(asPred, owl_path):
    """
    main function to process the OWL file and generate Datalog rules

    Args:
        asPred (bool): If True, use category as predicate. If False, use as object
    """
    g = Graph()
    g.parse(owl_path)
    rules = []

    #  unsure of how to use
    # variables = ['X', 'Y', 'Z', 'W', 'X1', 'X2']

    for s, p, o in g:
        process_triple(g, s, p, o, rules, asPred)

    for rule in rules:
        if not rule.endswith('.'):
            rule = rule + '.'
        print(rule)

    datalog_content = '\n'.join(rules)

    file_output_type = 'catPred' if asPred else 'predObj'
    # determine if the RDF file is a URL or a local file to get the name
    if urlparse(owl_path).scheme in ('http', 'https'):

        rdf_file_name = os.path.basename(urlparse(owl_path).path)
    else:
        rdf_file_name = os.path.basename(owl_path)

    if not rdf_file_name:
        rdf_file_name = re.sub(r'[<>:"/\\|?*]', '_', owl_path)

    output_file = f"{file_output_type}-{rdf_file_name}.txt"

    with open(output_file, 'w') as output:
        output.write(datalog_content)

    print(f"\n{len(rules)} rules converted")
    print(f"Datalog content has been written to {output_file}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Translate OWL Description Logic to LP Datalog")
    parser.add_argument("--catPred", action="store_true", help="Use category as predicate (default)")
    parser.add_argument("--predObj", action="store_true", help="Use category as object")
    parser.add_argument("--owl_path", default="Data/univ-bench.owl",
                        help="Path to the input OWL file (default: Data/univ-bench.owl)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # default to using catPred if no arguments
    asPred = not args.predObj

    main(asPred=asPred, owl_path=args.owl_path)
    # main(asPred=True, owl_path='http://xmlns.com/foaf/0.1/')

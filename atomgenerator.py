# Generates atoms for unification
import csv
from typing import Dict
import basictypes
import random
import numpy as np
import pandas as pd
import functools as ft
import argparse
import random
import kbparser
from copy import copy, deepcopy
from prints import print_progress_bar
from kbparser import parse_atom
from itertools import product

from vocab import Vocabulary


# generates 10 predicate objects
def gen_predicates(num_predicates=10):
    #    return [basictypes.Predicate( np.random.choice([0,1,2,3,4,5],p = [0.05,0.3,0.3,0.2,0.1,0.05]), "p" +str(x)) for x in range(num_predicates)]
    return [basictypes.Predicate(np.random.choice([0, 1, 2, 3, 4, 5], p=[0.0, 0.3, 0.3, 0.2, 0.1, 0.1]), "p" + str(x)) for x in range(num_predicates)]


# generates a specific number (num_atoms) of atoms with a list of predicates
# each atom has a random predicate taken from p, a list of random variables as arguments, and returns the list of atoms
def generate_atoms(num_atoms: int, p, num_constants: int = 100, num_variables: int = 10):
    constants = [basictypes.Constant("a" + str(x))
                 for x in range(num_constants)]
    variables = [basictypes.Variable("X" + str(x))
                 for x in range(num_variables)]
    predicates = p
    atoms = []
    for _ in range(num_atoms):
        p = deepcopy(predicates[random.randint(0, 9)])
        l = []
        for i in range(p.arity):
            if np.random.random() < 0.5:
                l.append(deepcopy(constants[np.random.randint(100)]))
            else:
                l.append(deepcopy(variables[np.random.randint(10)]))
        atoms.append(basictypes.Atom(p, l))
    return atoms

def generate_single_atom(vocab: Vocabulary):
    p = random.choice(vocab.predicates)
    l = []
    for _ in range(p.arity):
        if np.random.random() < 0.5:
            l.append(random.choice(vocab.constants))
        else:
            l.append(random.choice(vocab.variables))
    return basictypes.Atom(p, l)


#Generates a list of unique (non-repeating) atoms.
def generate_atoms_from_vocab(num_atoms: int, vocab: Vocabulary, use_legacy=False):
    atoms = set()
    num_atoms -= 1
    # while len(atoms) <= num_atoms:
    #     atoms.add(generate_single_atom(vocab))
    #     print_progress_bar(
    #         len(atoms), (num_atoms+1), length=20, suffix="Anchors generated")
    
    # Code for making sure there are no query repeats
    #TODO: Cahnge hard-coded query file names?
    open('test_queries.txt', 'a').close()
    open('train_queries.txt', 'a').close()
    parsed_queries = kbparser.parse_KB_file("test_queries.txt").rules
    test_queries = [query.head for query in parsed_queries]
    parsed_queries = kbparser.parse_KB_file("train_queries.txt").rules
    train_queries = [query.head for query in parsed_queries]

    if use_legacy:
        print("Using legacy atom generaton")
        atoms = list(atoms)
        while len(atoms) <= num_atoms:
            new_atom = generate_single_atom(vocab)
            atoms.append(new_atom)
            if len(atoms) % 100 == 0:
                print_progress_bar(
                    len(atoms), (num_atoms+1), length=20, suffix="Atoms generated")
    else:
        while len(atoms) <= num_atoms:
            new_atom = generate_single_atom(vocab)
            if (new_atom in test_queries) or (new_atom in train_queries):
                continue
            atoms.add(new_atom)
            if len(atoms) % 100 == 0:
                print_progress_bar(
                    len(atoms), (num_atoms+1), length=20, suffix="Anchors generated")
    print()
    return list(atoms)


# takes two atoms as arguments and tries to unify them with unification algorithm
# returns true if atoms can be unified, false if not
def unify_atoms(t1: basictypes.Atom, t2: basictypes.Atom):
    """Takes two atoms as arguments. Returns True if they can be unified
    with the unification algorithm and false otherwise."""
    if (t1.predicate != t2.predicate):
        return False
    else:
        if (t1.predicate.arity == 0):
            return True
        else:
            # We don't actually do anything with S, but other functions
            # rely on this function to return a boolean value.
            E = []
            S = []
            t1 = deepcopy(t1)
            t2 = deepcopy(t2)

            # E is a list of lists of the corresponding arguments of t1 and t2.
            # For example, if t1 is p(a, b) and t2 is p(X, b), then
            # E should denote [ [a, X], [b, b] ]
            for i in range(t1.predicate.arity):
                E.append([t1.arguments[i], t2.arguments[i]])

            # For each element of E, if possible apply the appropriate
            # substitution to all of E, otherwise return False
            while E:
                e = E.pop(0)
                if e[0] != e[1]:
                    if isinstance(e[0], basictypes.Variable):
                        for i, term_tuple in enumerate(E):
                            for j, term in enumerate(term_tuple):
                                if term == e[0]:
                                    E[i][j] = e[1]
                        S.append((e[0], e[1]))
                    elif isinstance(e[1], basictypes.Variable):
                        for i, term_tuple in enumerate(E):
                            for j, term in enumerate(term_tuple):
                                if term == e[1]:
                                    E[i][j] = e[0]
                        S.append((e[1], e[0]))
                    else:
                        return False
            return True

# takes atom as input; returns a list of variable names used


def get_vars(atom: basictypes.Atom):
    names = []
    for item in atom.arguments:
        names.append(item.name)
    return names

# assert unify_atoms(basictypes.Atom(basictypes.Predicate(2,"p"),[basictypes.Variable("v3"), basictypes.Constant("c")]), basictypes.Atom(basictypes.Predicate(2,"p"),[basictypes.Variable("v2"), basictypes.Variable("v1")]))
# assert not unify_atoms(basictypes.Atom(basictypes.Predicate(2,"p"),[basictypes.Variable("v1"), basictypes.Variable("v1")]), basictypes.Atom(basictypes.Predicate(2,"p"),[basictypes.Constant("c2"), basictypes.Constant("c1")]))
# v1 = basictypes.Variable("v1")
# v2 = basictypes.Variable("v2")
# v3 = basictypes.Variable("v3")
# v4 = basictypes.Variable("v4")
# c1 = basictypes.Constant("c1")
# c2 = basictypes.Constant("c2")
# c3 = basictypes.Constant("c3")
# c4 = basictypes.Constant("c4")
# p1 = basictypes.Predicate(2,"p1")
# p2 = basictypes.Predicate(2,"p2")
# p3 = basictypes.Predicate(0,"p3")
# atom1 = basictypes.Atom(p1,[v1,c1])
# assert unify_atoms(atom1, basictypes.Atom(p1,[v2,c1]))
# assert unify_atoms(atom1, basictypes.Atom(p1,[c2,c1]))
# assert unify_atoms(atom1, basictypes.Atom(p1,[c1,c1]))
# assert unify_atoms(atom1, basictypes.Atom(p1,[v2,v1]))
# assert unify_atoms(atom1, basictypes.Atom(p1,[v2,v2]))
# assert unify_atoms(atom1, basictypes.Atom(p1,[v1,v1]))
# assert not  unify_atoms(atom1, basictypes.Atom(p1,[v3,c2]))
# assert not unify_atoms(basictypes.Atom(p1,[v1,v1]), basictypes.Atom(p1,[c2,c1]))
# assert not unify_atoms(basictypes.Atom(p1,[v1,v1]), basictypes.Atom(p2,[v2,c1]))
# assert unify_atoms(basictypes.Atom(p3,[]), basictypes.Atom(p3,[]))
# assert not unify_atoms(basictypes.Atom(p1,[v1,v1]), basictypes.Atom(p3,[]))
# assert unify_atoms(atom1,atom1)


# takes iterations and number of atoms, generates sets of atoms and calculates the ratio of pairs of atoms that are identical
# repeats process according to desired number of iterations and returns the average ratio
def true_ratio(iterations, num_atoms, num_constants, num_variables):
    count = 0
    for i in range(iterations):
        scount = 0
        atoms = generate_atoms(num_atoms)
        for atom1 in atoms:
            for atom2 in atoms:
                if (unify_atoms(atom1, atom2)):
                    scount += 1
        div = num_atoms**2
        scount /= div
        count += scount
        print(i)
    count /= iterations
    return count

# Returns string, the base type of an atom for tallying up atom types. Used in gen_triplets
def typify_atom(atom:basictypes.Atom):
    atom_type = ""

    for i in range(atom.arity):
        main = atom.arguments[i]
        same_count = 0
        prefix = "d"
        main_type = "c" if isinstance(main, basictypes.Variable) else "v" # c for const, v for variable. c by default
        
        for j in range(atom.arity):
            if str(main) == str(atom.arguments[j]):
                same_count += 1
        if same_count != 1:
            prefix = "s"+str(same_count)+""
        
        atom_type += prefix+main_type+"_"
    
    return atom_type

def new_triplets(vocab: Vocabulary, anchors: list, triplet_path = "triplets.csv", triplet_set_size = 3):
    """
    Generates a set of 3 triplets for each anchor in a given list of anchor atoms

    :param vocab:
    :param anchors:
    :param triplet_path: Path to save the triplet file. When triplet_path=False, no triplet file is saved
    :param triplet_set_size: How large sets of triplets should be (Default 3 triplets per anchor)
    """
    triplets = []
    a_len = len(anchors)
    target_len = a_len*triplet_set_size # Used for printing stuff
    update_print = target_len * 0.005 # Also used for printing stuff

    for ind in range(a_len):
        anchor = anchors[ind]
        
        for i in range(triplet_set_size):
            args = anchor.arguments
            new_args = deepcopy(args)
            
            # generate Positive
            while (new_args == args):
                for i in range(len(args)):
                    if np.random.random() < 0.5: # 50% Chance an argument gets modified.
                        if isinstance(args[i], basictypes.Constant):
                            new_args[i] = random.choice(vocab.variables)
                        elif isinstance(args[i], basictypes.Variable):
                            if np.random.random() < 0.5:
                                new_args[i] = random.choice(vocab.constants)
                            else:
                                new_args[i] = random.choice(vocab.variables)
                
                    for j in range(len(args)): # Any duplicate arguments? Modify those as well.
                        if args[i] == args[j]:
                            new_args[j] = new_args[i]

                positive = basictypes.Atom(anchor.predicate, new_args)
                if not unify_atoms(anchor, positive): # No longer a positive example? Start from the original anchor.
                    # print(str(anchor)+" | "+str(positive))
                    new_args = deepcopy(args)
            
            positive = basictypes.Atom(anchor.predicate, new_args)

            # generate Negative
            negative = basictypes.Atom(anchor.predicate, args)
            new_args = deepcopy(args)
            all_variables = True
            for arg in args:
                if isinstance(arg, basictypes.Constant):
                    all_variables = False
                    break
            
            if all_variables:
                # Change predicate, since that's the only thing that makes it false.
                while unify_atoms(anchor, negative):
                    negative = generate_single_atom(vocab)
                
            else:
                while (new_args == args):
                    for i in range(len(args)):
                        if np.random.random() < 0.5:
                            if isinstance(args[i], basictypes.Constant):
                                new_args[i] = random.choice(vocab.constants)
                            elif isinstance(args[i], basictypes.Variable):
                                # Question for Jeff: Should we generate new variables even though it makes no difference in terms of unification.
                                # ---It might make a difference by making more possible negative examples.
                                new_args[i] = random.choice(vocab.variables)
                    
                    if unify_atoms(anchor, basictypes.Atom(anchor.predicate, new_args)): # Positive example? Try again from beginning.
                        new_args = deepcopy(args)
            
                negative = basictypes.Atom(anchor.predicate, new_args)
            
            triplets.append([anchor, positive, negative])
        
        current_trips = (ind+1)*triplet_set_size
        if current_trips % update_print == 0:
            print_progress_bar(
                current_trips, target_len, length=20, suffix="Triplets generated")

    print()
    if triplet_path:
        with open('triplets.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            for triple in triplets:
                writer.writerow([str(triple[0]), str(triple[1]), str(triple[2])]) # [anchor, positive, negative]
    
    return triplets
        

def gen_triplets(atoms, min_triplets, vocab: Vocabulary):
    """
    DEPRECATED, used in --use_legacy_embeddings:
    Generates positive and negative triplets from a set of atoms.
    Creates positive triplets if two atoms can be unified, and negative if not. It utilizes unify_atoms for this step.
    Returns two dictionaries of the positive and negative triplets.
    """
    num_atoms = len(atoms)
    positives = {}
    negatives = {}
    count = 0
    for atom1 in atoms:
        count = 0
        for key in positives:
            count += len(positives[key])
        print_progress_bar(count, min_triplets, length=20, suffix="pairs")
        if count >= min_triplets:
            break
        for atom2 in atoms:
            if (unify_atoms(atom1, atom2)):
                hash_a1 = str(atom1) # Note: these aren't the hashes of the atom, but the literal atom string e.g. "mother(sally, molly)"
                hash_a2 = str(atom2)
                if (hash_a1 not in positives):
                    positives[hash_a1] = {hash_a1: atom1, hash_a2: atom2}
                else:
                    positives[hash_a1][hash_a2] = atom2

                if (hash_a2 not in positives):
                    positives[hash_a2] = {hash_a2: atom2, hash_a1: atom1}
                else:
                    positives[hash_a2][hash_a1] = atom1
    print()

# Code to check whether the positives actually unify or not.
# The fixed unify_atoms() seems to solve this problem.
#    for key in positives:
#        for key2 in positives[key]:
#            a = positives[key][key] # Should be the anchor
#            b = positives[key][key2]
#            if not unify_atoms(a, b):
#                print(a, b, "do not unify")
#                if unify_atoms(b, a):
#                    print("Bug in unification algorithm")
#                if unify_atoms(a, b):
#                    print("What the heck?")
#                exit(1)
    triplet_examples = {}
    for key in positives:
        positive_example = None
        for k in positives[key]:
            if k == key:
                continue
            positive_example = k
        triplet_examples[key] = [key, positive_example] # [anchor, positive, negative]

    current = 0
    for key in positives:
        a = positives[key][key]
        for i in range(len(positives[key])):
            not_added = True
            while not_added:
                neg = deepcopy(a)
                give_up_percent = 0.00
                while (unify_atoms(a, neg)):
                    # could make it more likely after it keeps trying and start lower
                    if (np.random.random() <= give_up_percent):
                        while (unify_atoms(a, neg)):
                            neg = atoms[np.random.randint(num_atoms)]
                        break
                    give_up_percent += 0.05
                    generate_negative(neg, vocab)
                hash_a1 = str(neg)
                hash_a2 = key
                if hash_a1 in negatives:
                    if (hash_a2 not in negatives[hash_a1]):
                        negatives[hash_a1][hash_a2] = a
                        if (hash_a2 in negatives):
                            negatives[hash_a2][hash_a1] = neg
                        else:
                            negatives[hash_a2] = {hash_a1: neg}
                        not_added = False
                else:
                    negatives[hash_a1] = {hash_a2: a}
                    if (hash_a2 in negatives):
                        negatives[hash_a2][hash_a1] = neg
                    else:
                        negatives[hash_a2] = {hash_a1: a}
                    not_added = False

                triplet_examples[hash_a2].append(hash_a1)
        current += 1
        print_progress_bar(current, len(positives),
                           length=20, suffix="negatives")
    print()

    # Note: Triplets are always saved by default. No option for this yet.
    with open('triplets.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for key in triplet_examples:
            triple = triplet_examples[key] # current triplet
            writer.writerow([triple[0], triple[1], triple[2]]) # [anchor, positive, negative]

    return [positives, negatives]

# called by gen_triplets to generate negative example by modifying an argument


def generate_negative(atom: basictypes.Atom, vocab: Vocabulary):
    # could add section to change predicate, but the system would likely generate those situations in the other method most of the time
    # keeping same arity probably makes "harder" negatives
    if (atom.predicate.arity == 0):
        pass
    else:
        if (np.random.random() < 0.5):
            atom.arguments[np.random.randint(
                atom.predicate.arity)] = random.choice(vocab.constants)
        else:
            atom.arguments[np.random.randint(
                atom.predicate.arity)] = random.choice(vocab.variables)

# generates a new variable with a randomly generated name beginning with X and ending with a number 0-9


def generate_var():
    return basictypes.Variable("X"+str(np.random.randint(10)))

# generates a new constant with a randomly generated name beginning with a and ending with a number 0-9


def generate_const():
    return basictypes.Constant("a"+str(np.random.randint(10)))


# def onehotencoding(atom: basictypes.Atom, num_pred: int = 10, num_var: int = 10, num_const: int = 100):
#     """ Takes an atom and returns a one-hot encoding of the predicate and arguments.
#     Replaced by Vocab.oneHotEncoding
#
#     :param atom: The atom to convert
#     :param num_pred: The total number of predicates in the KB
#     :param num_var: The total number of variables in the KB
#     :param num_const: The total number of constants in the KB
#     :return: The one-hot encoding of the atom
#     """
#     total_terms = num_var+num_const
#     encoding = [0 for x in range(num_pred+total_terms*5)]
#     pred = int(atom.predicate.name[1:])
#     encoding[pred] = 1
#     start = -(total_terms - num_pred)
#     for arg in atom.arguments:
#         start += total_terms
#         if (isinstance(arg, basictypes.Constant)):
#             move = 10
#         else:
#             move = 0
#         encoding[start+move+int(arg.name[1:])] = 1
#     return encoding

# Take a one-hot encoding of an atom and return the atom representation


def reverse_encoding(encoding: list):
    """Takes a one-hot encoding of an atom as an input and returns the corresponding atom.
       This function assumes that the encoding was generated by onehotencoding() and thereby
       has 10 predicates, 10 variables, 100 constants, and 5 possible arguments."""
    predicates_end = 9
    var1_end = 19
    const1_end = 119
    var2_end = 129
    const2_end = 229
    var3_end = 239
    const3_end = 339
    var4_end = 349
    const4_end = 449
    var5_end = 459
    const5_end = 559

    indices = [i for i, j in enumerate(encoding) if j == 1]
    arg_list = []
    p = basictypes.Predicate(None, None)

    if len(indices) >= 1:
        p.arity = len(indices)-1
        p.name = f"p{indices[0]}"

    if len(indices) >= 2:
        if indices[1] <= var1_end:
            arg_list.append(basictypes.Variable(
                f"X{indices[1] - (predicates_end+1)}"))
        else:
            arg_list.append(basictypes.Constant(
                f"a{indices[1] - (var1_end+1)}"))

    if len(indices) >= 3:
        if indices[2] <= var2_end:
            arg_list.append(basictypes.Variable(
                f"X{indices[2] - (const1_end+1)}"))
        else:
            arg_list.append(basictypes.Constant(
                f"a{indices[2] - (var2_end+1)}"))
    if len(indices) >= 4:
        if indices[3] <= var3_end:
            arg_list.append(basictypes.Variable(
                f"X{indices[3] - (const2_end+1)}"))
        else:
            arg_list.append(basictypes.Constant(
                f"a{indices[3] - (var3_end+1)}"))
    if len(indices) >= 5:
        if indices[4] <= var4_end:
            arg_list.append(basictypes.Variable(
                f"X{indices[4] - (const3_end+1)}"))
        else:
            arg_list.append(basictypes.Constant(
                f"a{indices[4] - (var4_end+1)}"))
    if len(indices) == 6:
        if indices[5] <= var5_end:
            arg_list.append(basictypes.Variable(
                f"X{indices[5] - (const4_end+1)}"))
        else:
            arg_list.append(basictypes.Constant(
                f"a{indices[5] - (var5_end+1)}"))

    a = basictypes.Atom(p, arg_list)
    return a

# takes positives and negatives dictionaries and takes the anchor, pos, and neg value, encoding them into one hot vectors
# returns separate lists for anchors, positives, negatives, and anchor names

def triplet_encodings(vocab: Vocabulary, positives: Dict, negatives: Dict):
    """
    DEPRECATED, Used in --use_legacy_embeddings:
    Takes positives and negatives dictionaries and then builds triplets from them.
    Each triplet consists of an anchor, pos, and neg value, and is encoded them into
    one hot vectors. Returns separate lists for anchors, positives, negatives, and anchor names

    :param vocab:
    :param positives:
    :param negatives:
    :return: Three lists of torch.Tensor: anchor, pos, neg, and one list of strings: predicate names
    """
    anchors, pos, neg = [], [], []
    pos_dict = deepcopy(positives)
    neg_dict = deepcopy(negatives)

    for i, key in enumerate(pos_dict):
        while (pos_dict[key] and neg_dict[key]):
            #Spencer: Changed one hot encoding to encode atom
            anchors.append(vocab.encodeAtom(pos_dict[key][key]))
            pos.append(vocab.encodeAtom(pos_dict[key].popitem()[1]))
            neg.append(vocab.encodeAtom(neg_dict[key].popitem()[1]))
        print_progress_bar(
            i+1, len(pos_dict), length=20, suffix="triplets")

    print()

    return anchors, pos, neg

def encode_triplets(vocab: Vocabulary, triplets: list):
    """
    One hot encodes & returns given anchors, positives, and negatives.
    """
    anc, pos, neg = [], [], []
    num_trips = len(triplets)

    for i in range(num_trips):
        triplet = triplets[i]
        # Spencer's new code + added param
        anc.append(vocab.encodeAtom(triplet[0]))
        pos.append(vocab.encodeAtom(triplet[1]))
        neg.append(vocab.encodeAtom(triplet[2]))

        print_progress_bar(
            i+1, num_trips, length=20, suffix="Triplet encodings")
    
    print()
    return anc, pos, neg

def extract_triplets(vocab: Vocabulary, triplet_path="triplets.csv"):
    df = pd.read_csv(triplet_path, delimiter='\t')
    anchors, pos, neg = [], [], []
    
    for i, row in df.iterrows():
        anchor_atom = parse_atom(row.iloc[0])
        pos_atom = parse_atom(row.iloc[1])
        neg_atom = parse_atom(row.iloc[2])

        # Spencer's new code + added param
        anchors.append(vocab.encodeAtom(anchor_atom))
        pos.append(vocab.encodeAtom(pos_atom))
        neg.append(vocab.encodeAtom(neg_atom))

        print_progress_bar(
            i+1, len(df.index), length=20, suffix="Triplet extraction")
    
    print()
    return anchors, pos, neg
    

def create_unity_embeddings(
        vocab: Vocabulary, 
        a_path, 
        p_path, 
        n_path, 
        num_triplets=70000, 
        save=False, 
        use_triplet_file: str = False,
        use_legacy_embeddings = False,
        set_size = 3,
        use_hash_encodings=False):
    """ Creates unity embeddings from a list of predicates by generating atoms, generating triplets,
    calling triplet_encodings, and creating dataframes from the encodings, saving them to respective files.
    The results are saved as CSV files.
    TODO: CSV is not the most efficient save format for the resulting vectors!
    Also, we need a human readable version of the output file

    Note: Triplets are automatically saved by default, will discuss about this with Jeff

    :param vocab:
    :param a_path:
    :param p_path:
    :param n_path:
    :param num_triplets:
    :param save: If true, saves one hot unity embeddings into a_path, p_path, and n_path
    :param use_saved_trips: Path of saved triplets to use
    :param use_legacy_embeddings: Use the previous version of the embedding model
    :param set_size:
    :return:
    """
    train_trips = None
    vocab.set_use_hash_const(use_hash_encodings)
    if not use_legacy_embeddings:
        if use_triplet_file:
            print("Existing triplet file, skipping triplet generation (--triplet_path)")
            train_trips = extract_triplets(vocab, use_triplet_file)
        else:
            #TODO: Make an option for the set_size
            atoms = generate_atoms_from_vocab(int(num_triplets/set_size), vocab) # Divided by 3 because we end up generating 3 triples for the same anchor.
            trips = new_triplets(vocab, atoms, triplet_set_size = set_size)
            train_trips = encode_triplets(vocab, trips)
            del atoms
            del trips
    else:
        print("Using legacy embeddings (--use_legacy_embeddings)")
        atoms = generate_atoms_from_vocab(int(num_triplets/15), vocab, use_legacy=True)
        trips = gen_triplets(atoms, num_triplets, vocab)
        train_trips = triplet_encodings(vocab, trips[0], trips[1])
        del atoms
        del trips
        for i in range(len(train_trips[0])):
            if not i < len(train_trips[1]):
                break
            if (train_trips[0][i] == train_trips[1][i]):
                train_trips[0].pop(i)
                train_trips[1].pop(i)
                train_trips[2].pop(i)
        
    if save:
        print("Saving anchor, positive, negative to csv...")
        df = pd.DataFrame(train_trips[0])
        df.to_csv(a_path, index=False)
        xf = pd.DataFrame(train_trips[1])
        xf.to_csv(p_path, index=False)
        cf = pd.DataFrame(train_trips[2])
        cf.to_csv(n_path, index=False)

    return train_trips

# DEPRECATED: Still uses gen_triplets which is deprecated.
# tests unify_atoms using assert statements
# generates and encodes triples, removing duplicates from the encoded triples
# stores encoded triples in files
if __name__ == "__main__":
    assert isinstance(unify_atoms(basictypes.Atom(basictypes.Predicate(2, "parent"), [basictypes.Constant("lisa"), basictypes.Constant(
        "lisa")]), basictypes.Atom(basictypes.Predicate(2, "parent"), [basictypes.Constant("homer"), basictypes.Constant("lisa")])), bool)
    print("pass")
    print(unify_atoms(basictypes.Atom(basictypes.Predicate(2, "parent"), [basictypes.Constant("lisa"), basictypes.Constant(
        "lisa")]), basictypes.Atom(basictypes.Predicate(2, "parent"), [basictypes.Constant("homer"), basictypes.Constant("lisa")])))
# print(atom_encodings)
    assert unify_atoms(basictypes.Atom(basictypes.Predicate(1, "p8"), [basictypes.Variable(
        "x1")]), basictypes.Atom(basictypes.Predicate(1, "p8"), [basictypes.Variable("x0")]))

    aparser = argparse.ArgumentParser()
    aparser.add_argument("-c", "--constants", type=int,
                         default=100, help="Number of constants in symbol set")
    aparser.add_argument("-v", "--variables", type=int,
                         default=10, help="Number of variables in symbol set")
    aparser.add_argument("-p", "--predicates", type=int,
                         default=10, help="Number of predicates in symbol set")

    args = aparser.parse_args()

    trips = gen_triplets(generate_atoms(4000, gen_predicates(
        args.predicates), args.constants, args.variables), 70000)
    print("train generated")
    train_trips = triplet_encodings(trips[0], trips[1])
    print("train encoded")
    for i in range(len(train_trips[0])):
        if not i < len(train_trips[1]):
            break
        if (train_trips[0][i] == train_trips[1][i]):
            train_trips[0].pop(i)
            train_trips[1].pop(i)
            train_trips[2].pop(i)
            train_trips[3].pop(i)

    print("train cleaned")

    df = pd.DataFrame(train_trips[0])
    df.to_csv("train_anchors.csv", index=False)
    xf = pd.DataFrame(train_trips[1])
    xf.to_csv("train_positives.csv", index=False)
    cf = pd.DataFrame(train_trips[2])
    cf.to_csv("train_negatives.csv", index=False)
    pf = pd.DataFrame(train_trips[3])
    pf.to_csv("train_names.csv", index=False)


# takes in two atoms and returns encoded representations of the atoms
def encode_two_atoms(a1: basictypes.Atom, a2: basictypes.Atom):
    a1_encoding = [1, 0]
    a2_encoding = []
    a1_args = [0 for i in range(a1.predicate.arity)]
    a2_args = [0 for i in range(a2.predicate.arity)]
    const_num = 0
    var_num = 0
    a1_dict = {}
    a2_dict = {}

    for i in range(len(a1_args)):
        if (a1.arguments[i] not in a1_dict):
            if (isinstance(a1.arguments[i], basictypes.Constant)):
                a2_dict[a1.arguments[i]] = ('c', const_num)
                a1_dict[a1.arguments[i]] = ('c', const_num)
                const_num += 1
            else:
                a1_dict[a1.arguments[i]] = ('v', var_num)
                var_num += 1
        a1_args[i] = a1_dict[a1.arguments[i]]

    for i in range(len(a2_args)):
        if (a2.arguments[i] not in a2_dict):
            if (isinstance(a2.arguments[i], basictypes.Constant)):
                a2_dict[a2.arguments[i]] = ('c', const_num)
                a1_dict[a2.arguments[i]] = ('c', const_num)
                const_num += 1
            else:
                a2_dict[a2.arguments[i]] = ('v', var_num)
                var_num += 1
        a2_args[i] = a2_dict[a2.arguments[i]]

    if (a1.predicate == a2.predicate):
        a2_encoding.extend([1, 0])
    else:
        a2_encoding.extend([1, 0])

import atomgenerator
import argparse
import pandas as pd
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--anchors", help="Training anchors path")
    parser.add_argument("-p", "--positives", help="Training positives path")
    parser.add_argument("-n", "--negatives", help="Training negatives path")
    parser.add_argument("-v", "--verbose", type=int, default=0,
                        help="0 for summary, 1 for failures, \
                        2 for both successes and failures")

    args = parser.parse_args()

    if args.anchors is None:
        sys.exit(1)

    anchors = pd.read_csv(args.anchors).to_numpy()
    if args.positives is not None:
        print("Checking anchors and positives")
        positives = pd.read_csv(args.positives).to_numpy()

        total = 0
        unifies = 0

        for i, j in zip(anchors, positives):
            i_atom = atomgenerator.reverse_encoding(i)
            j_atom = atomgenerator.reverse_encoding(j)
            if atomgenerator.unify_atoms(i_atom, j_atom):
                if args.verbose >= 2:
                    print(i_atom, j_atom, "unifies")
                total += 1
                unifies += 1
            else:
                if args.verbose >= 1:
                    print(i_atom, j_atom, "fails to unify")
                total += 1

            if atomgenerator.unify_atoms(i_atom, j_atom) != atomgenerator.unify_atoms(i_atom, j_atom):
                print("Bug in unification algorithm")
        print(f"{unifies}/{total} unify")

    if args.negatives is not None:
        print("Checking anchors and negatives")
        negatives = pd.read_csv(args.negatives).to_numpy()

        total = 0
        unifies = 0

        for i, j in zip(anchors, negatives):
            i_atom = atomgenerator.reverse_encoding(i)
            j_atom = atomgenerator.reverse_encoding(j)

            if not atomgenerator.unify_atoms(i_atom, j_atom):
                if args.verbose >= 2:
                    print(i_atom, j_atom, "fails to unify")
                total += 1
            else:
                if args.verbose >= 1:
                    print(i_atom, j_atom, "unifies")
                unifies += 1
                total += 1

            if atomgenerator.unify_atoms(i_atom, j_atom) != atomgenerator.unify_atoms(i_atom, j_atom):
                print("Bug in unification algorithm")
        print(f"{unifies}/{total} unify")

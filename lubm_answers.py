from basictypes import Constant, Variable
from kbparser import parse_KB_file
from vocab import Vocabulary

answer_path = "lubm1-answers/"
def iri_to_constant(iri:str) -> str:
    """Converts an IRI to a Datalog constant.
    """
    sym = iri

    # There is a bug in the current LUBM data where the department number is always being changed to 0!
    # TODO: fix this when the LUBM data is fixed
    edu_start = sym.find(".edu/")
    if edu_start >= 0:
        sym = "d0u0_" + sym[edu_start+5:]
    # sym = sym.replace("http://www.Department0Universty0.edu","d0u0")

    # TODO: change first letter to lower-case for values of name! i.e., anything that doesn't start with http

    # handle IRIs of departments
    sym = sym.replace("http://www.Department","department")
    sym = sym.replace(".University","university", 1)
    sym = sym.replace(".edu","")

    sym = sym.replace("/", "_")
    return sym


def read_answer_file(file_path:str, pattern:str, vars:list[str]) -> list[str]:
    facts = []
    with open(file_path, mode='r') as f:
        lines = f.readlines()
        header = lines[0].split()
        if len(header) != len(vars):
            print("Warning: different number of file bindings than in query " + pattern)

        lines = lines[1:]                  # remove first row (the variable names)
        if len(lines) > 80:                # truncate very long answer sets
            lines = lines[0:79]
        for line in lines:
            line = line.strip()
            bindings = line.split()
            fact = pattern
            for i in range(len(vars)):
                bind = iri_to_constant(bindings[i])
                fact = fact.replace(vars[i], bind)
            facts.append(fact)
    return facts


if __name__ == "__main__":
    patterns = [("q1(X).", ["X"], "answers_query1.txt"),
                ("q3(X).", ["X"], "answers_query3.txt"),
                ("q4(X, Y).", ["X","Y"], "answers_query4.txt"),
                ("q5(X).", ["X"], "answers_query5.txt"),
                ("q6(X).", ["X"], "answers_query6.txt"),
                ("q7(X, Y).", ["X","Y"], "answers_query7.txt"),
                ("q8(X, Y).", ["X","Y"], "answers_query8.txt"),
                ("q10(X).",  ["X"], "answers_query10.txt"),
                ("q11(X).", ["X"], "answers_query11.txt"),
                ("q12(X, Y).", ["X","Y"], "answers_query12.txt"),
                ("q13(X).",  ["X"], "answers_query13.txt"),
                ("q14(X).",  ["X"], "answers_query14.txt")]

    facts = []
    for p in patterns:
        new_facts = read_answer_file(answer_path + p[2], p[0], p[1])
        facts = facts + new_facts

    with open("lubmq-facts.txt", mode='w') as outfile:
        print("Writing to " + outfile.name)
        for fact in facts:
            outfile.write(fact + "\n")
        outfile.close()
    print("Done!")

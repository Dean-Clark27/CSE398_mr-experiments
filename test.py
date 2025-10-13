from atomgenerator import gen_predicates
import kbparser
import reasoner
import knowledgebase
from atomgenerator import gen_predicates
import kbencoder

# prompts user to input a path to a KB file, then allows the user to query
# the KB using forward and backward chaining, printing out the results

kb_path = input("Input knowledge base path: ")
KB = kbparser.parse_KB_file(kb_path)
while True:  
    #KB = parser.parse_KB_file(kb_path)

    
    print()
    for fact in reasoner.forwardchain(KB):
        print(fact)

    
    query = input("Type query: ")
    try:
        query = kbparser.parse_atom(query)
    except:
        print("parse failed")
        continue
    path = reasoner.Path(query, None, None, 0)
    examples = set()
    answers = kbencoder.backwardchain(KB, path, reasoner.MaxDepth(99999), examples)
    for x in answers:
        print('true')
        for key in x:
                print(f"var {key}")
                print(f"sub: {x[key]}")


    
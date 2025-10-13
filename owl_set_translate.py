import os
import re
from urllib.parse import urlparse
# import unidecode

import validators
from rdflib import Graph


def extract_upper_and_numbers(string):
    return ''.join(filter(lambda char: char.isupper(), string))


def remove_first_last_from_list(input_list):
    return input_list[1:-1] if len(input_list) >= 2 else input_list


def lowercase_first_char(input_string):
    if input_string[0].isdigit():     # if string starts with a number, prepend character 'n'
        input_string = "n" + input_string
    return input_string[0].lower() + input_string[1:] if input_string else input_string


def clean_string(s):
    # remove special characters, keep numbers
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
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


def parse_url(url):
    if not validators.url(url):
        return clean_string(url)

    parts = urlparse(url)
    path = parts.path.strip('/')

    if parts.fragment:
        return clean_string(lowercase_first_char(parts.fragment))
    elif parts.netloc and parts.path:
        netloc = clean_string(extract_upper_and_numbers(parts.netloc))
        path = "_".join([x for x in parts.path.split("/") if x])
        return clean_string(lowercase_first_char("_".join([netloc, path])))
    elif parts.netloc:
        netloc = "".join(remove_first_last_from_list(parts.netloc.split(".")))
        return clean_string(lowercase_first_char(netloc))
    else:
        return clean_string(lowercase_first_char(url))

    # if path:
    #     return clean_string(path.split('/')[-1])
    # elif parts.fragment:
    #     return clean_string(parts.fragment)
    # else:
    #     return clean_string(parts.netloc.split('.')[-2])  # Use domain name without TLD


def is_email(input_string):
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, input_string))


def contains_non_english(string):
    return any(ord(char) > 127 for char in string)


def main(input_file, format, output_file, is_unary=False):
    print("File: ", input_file)
    g = Graph()
    g.parse(input_file, format=format)

    datalog_facts = []
    ignore_list = {"telephone", "ontology", "imports", "comment", "label"}

    for index, (subj, pred, obj) in enumerate(g):
        # if index >= 1000:
        #     break

        subj_str = parse_url(str(subj))
        pred_str = parse_url(str(pred))
        obj_str = parse_url(str(obj))

        # TODO: some facts parse empty objects, figure out why but pass for now
        if not subj_str or not pred_str or not obj_str:
            continue

        if pred_str in ignore_list or subj_str in ignore_list or obj_str in ignore_list:
            continue

        # TODO: Do we really want to remove all email addresses?
        if is_email(str(subj)) or is_email(str(pred)) or is_email(str(obj)):
            continue

        if contains_non_english(subj_str) or contains_non_english(pred_str) or contains_non_english(obj_str):
            continue

        if is_unary and pred_str == "type":
            datalog_fact = f"{obj_str}({subj_str})."
        else:
            datalog_fact = f"{pred_str}({subj_str}, {obj_str})."

        datalog_facts.append(datalog_fact)

    datalog_content = '\n'.join(datalog_facts)

    with open(output_file, 'w') as output:
        output.write(datalog_content)

    print(datalog_content)
    print(f"Datalog content has been written to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert RDF data to Datalog format")
    parser.add_argument("--input_file", help="Path to the input RDF file")
    parser.add_argument("--format", help="format of the RDF file", default="application/rdf+xml")
    parser.add_argument("--output_file", help="Path to the output Datalog file")
    parser.add_argument("--is_unary", action="store_true", help="Use unary predicates for 'type'")

    args = parser.parse_args()

    main(args.input_file, args.format, args.output_file, args.is_unary)
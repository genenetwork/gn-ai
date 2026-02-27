import json
import os
import string
from copy import copy

dir = os.getenv("DIR")
if dir is None:
    raise ValueError("DIR not specified")

collection = {}

files = [os.path.join(dir, file) for file in os.listdir(dir) if "ttl" in file]


def clean(text) -> str:
    return text.strip(string.punctuation).strip()


for file in files:
    with open(file) as f:
        lines = f.readlines()
        for ind, line in enumerate(lines):
            line = line.strip()
            if len(line) > 1:
                contents = line.split(" ")
                key = clean(contents[0])
                value = " ".join(clean(content) for content in contents[1:])
                if key not in collection:
                    collection[key] = [value]
                else:
                    if value not in collection[key]:
                        collection[key].append(value)

with open(f"{dir}aggr_rdf.txt", "w") as new:
    new.write(json.dumps(collection))

import json
import pdb

from pathlib import Path
from aletheia.utils import read_file

FOLDER = Path("/mnt/private-share/speechDatabases/lrs3/lrs3_v0.4/test")

def parse_line(line):
    if line.startswith("Text:"):
        words = line.split()
        return " ".join(words[1:])
    else:
        return None


data = []
for dir in FOLDER.iterdir():
    for file in dir.iterdir():
        text = read_file(str(file), parse_line)
        text = [t for t in text if t is not None]
        datum = {
            "text": " ".join(text),
            "file": str(file),
        }
        data.append(datum)


with open("output/lrs3-test-sentences.json", "w") as f:
    json.dump(data, f, indent=4)

with open("output/lrs3-test-sentences.txt", "w") as f:
    for datum in data:
        f.write(datum["text"] + "\n")
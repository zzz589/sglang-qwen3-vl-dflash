"""Compare greedy DFlash2-VL and target-only output on identical real requests."""

import json
import sys
from urllib.request import Request, urlopen


data_path, model_path = sys.argv[1:3]
count = int(sys.argv[3]) if len(sys.argv) > 3 else 5
draft_port = int(sys.argv[4]) if len(sys.argv) > 4 else 31055
target_port = int(sys.argv[5]) if len(sys.argv) > 5 else 31056
summary_only = len(sys.argv) > 6 and sys.argv[6] == "summary"


def generate(port, payload):
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=240) as response:
        result = json.load(response)
    return result["choices"][0]["message"]["content"]


matches = 0
with open(data_path, encoding="utf-8") as stream:
    for index, line in enumerate(stream):
        if index >= count:
            break
        row = json.loads(line)
        payload = {
            "model": model_path,
            "messages": row["conversations"][:-1],
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
        }
        draft = generate(draft_port, payload)
        target = generate(target_port, payload)
        equal = draft == target
        matches += equal
        if not summary_only or not equal:
            print(json.dumps({"id": row["id"], "equal": equal, "draft": draft, "target": target}, ensure_ascii=False), flush=True)
print(json.dumps({"count": count, "exact_matches": matches, "match_fraction": matches / count}), flush=True)

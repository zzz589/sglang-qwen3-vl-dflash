"""Check whether target-only and draft outputs are deterministic for suspect rows."""

import json
import sys
from urllib.request import Request, urlopen


data_path, model_path = sys.argv[1:3]
ids = set(sys.argv[3].split(","))
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 3


def generate(port, payload):
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=240) as response:
        result = json.load(response)
    return result["choices"][0]["message"]["content"]


with open(data_path, encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        if row["id"] not in ids:
            continue
        payload = {
            "model": model_path,
            "messages": row["conversations"][:-1],
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
        }
        for index in range(rounds):
            print(json.dumps({
                "id": row["id"],
                "round": index,
                "draft": generate(31055, payload),
                "target": generate(31056, payload),
            }, ensure_ascii=False), flush=True)

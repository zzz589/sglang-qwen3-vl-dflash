"""Inspect the first differing token in speculative versus target-only replies."""

import json
import sys
from urllib.request import Request, urlopen


data_path, model_path = sys.argv[1:3]
ids = set(sys.argv[3].split(","))


def generate(port, payload):
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=240) as response:
        result = json.load(response)
    return result["choices"][0]


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
            "logprobs": True,
            "top_logprobs": 5,
            "stream": False,
        }
        draft = generate(31055, payload)
        target = generate(31056, payload)
        a = draft["logprobs"]["content"]
        b = target["logprobs"]["content"]
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x["token"] != y["token"]), None)
        if len(sys.argv) > 4 and sys.argv[4] == "summary":
            target_top = b[first]["top_logprobs"] if first is not None else []
            draft_top = a[first]["top_logprobs"] if first is not None else []
            print(json.dumps({
                "id": row["id"],
                "first_different_token": first,
                "draft_token": a[first]["token"] if first is not None else None,
                "target_token": b[first]["token"] if first is not None else None,
                "target_top2_logit_gap": target_top[0]["logprob"] - target_top[1]["logprob"] if len(target_top) > 1 else None,
                "draft_top2_logit_gap": draft_top[0]["logprob"] - draft_top[1]["logprob"] if len(draft_top) > 1 else None,
            }, ensure_ascii=False), flush=True)
            continue
        print(json.dumps({
            "id": row["id"],
            "first_different_token": first,
            "draft_text": draft["message"]["content"],
            "target_text": target["message"]["content"],
            "draft_before": a[max(0, first - 2): first + 2] if first is not None else [],
            "target_before": b[max(0, first - 2): first + 2] if first is not None else [],
        }, ensure_ascii=False), flush=True)

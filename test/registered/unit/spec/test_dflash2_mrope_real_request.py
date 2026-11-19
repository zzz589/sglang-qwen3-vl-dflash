"""Exercise the isolated DFlash2-VL server with a real regenerated image row."""

import json
import sys
from urllib.request import Request, urlopen


count = int(sys.argv[3]) if len(sys.argv) > 3 else 1
port = int(sys.argv[4]) if len(sys.argv) > 4 else 31055
with open(sys.argv[1], encoding="utf-8") as stream:
    for index, line in enumerate(stream):
        if index >= count:
            break
        row = json.loads(line)
        payload = {
            "model": sys.argv[2],
            "messages": row["conversations"][:-1],
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
        }
        request = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=240) as response:
            result = json.load(response)
        assert result.get("choices"), result
        print(json.dumps({
            "id": row["id"],
            "text": result["choices"][0]["message"]["content"],
            "usage": result.get("usage"),
        }, ensure_ascii=False), flush=True)

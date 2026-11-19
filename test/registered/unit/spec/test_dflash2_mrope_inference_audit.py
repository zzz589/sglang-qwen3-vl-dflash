"""Focused checks for DFlash2-VL inference position handling."""

from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2


def main():
    worker = object.__new__(DFlashWorkerV2)
    mm = SimpleNamespace(
        mrope_positions=torch.tensor(
            [[0, 1, 2, 3, 4], [0, 1, 20, 21, 22], [0, 1, 30, 31, 32]],
            dtype=torch.long,
        ),
        mrope_position_delta=torch.tensor(17),
    )
    batch = SimpleNamespace(multimodal_inputs=[mm, None])
    prompt = worker._prompt_mrope_positions(batch, [1, 0], [3, 2], "cpu")
    expected = torch.tensor(
        [[1, 2, 3, 0, 1], [1, 20, 21, 0, 1], [1, 30, 31, 0, 1]],
        dtype=torch.long,
    )
    torch.testing.assert_close(prompt, expected)
    blocks = torch.tensor([[5, 6, 7], [2, 3, 4]], dtype=torch.long)
    generated = worker._block_mrope_positions(batch, blocks)
    expected_generated = torch.tensor([[22, 23, 24, 2, 3, 4]] * 3)
    torch.testing.assert_close(generated, expected_generated)
    print("DFlash2 mRoPE position mapping: PASS")


if __name__ == "__main__":
    main()

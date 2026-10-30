"""Process-wide patch, picked up by every Ray worker because the project root is on PYTHONPATH.

verl 0.9.1 falls back to `logprobs_from_logits_v2` when flash-attn is not installed (our case: no flash-attn
wheel for torch 2.11). That function runs `log_softmax` (bf16) or `logsumexp` (fp32) once per token row, i.e.
~850k single-block kernel launches per training step over the 248k-entry Qwen3.5 vocabulary; it was ~60% of
the actor's GPU time. The replacement does the same math (same dtype, log_softmax + gather / logsumexp) on
chunks of rows. verl itself is left unmodified; the module is patched right after it is imported.
"""

import importlib.abc
import importlib.util
import sys

_TARGET = "verl.utils.torch_functional"
_CHUNK_ROWS = 1024


def _logprobs_from_logits_chunked(logits, labels):
    import torch
    import torch.nn.functional as F

    shape = logits.shape[:-1]
    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.reshape(-1)
    out = []
    for start in range(0, logits.shape[0], _CHUNK_ROWS):
        chunk, chunk_labels = logits[start:start + _CHUNK_ROWS], labels[start:start + _CHUNK_ROWS]
        if logits.dtype in (torch.float32, torch.float64):
            lp = chunk.gather(-1, chunk_labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(chunk, dim=-1)
        else:
            lp = F.log_softmax(chunk, dim=-1).gather(-1, chunk_labels.unsqueeze(-1)).squeeze(-1)
        out.append(lp)
    return torch.cat(out).view(shape)


def _patch(module):
    module.logprobs_from_logits_v2 = _logprobs_from_logits_chunked


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None:
            return None
        loader = spec.loader
        original_exec = loader.exec_module

        def exec_module(module):
            original_exec(module)
            _patch(module)

        loader.exec_module = exec_module
        return spec


if _TARGET in sys.modules:
    _patch(sys.modules[_TARGET])
else:
    sys.meta_path.insert(0, _Finder())

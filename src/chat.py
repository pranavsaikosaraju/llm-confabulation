"""Chat-model utilities: conversations, answer logits, generation and steering.

A conversation is a list of {"role", "content"} dicts. Three kinds of reads:

* `choice_logits`  : next-token preference between two answer words (one forward pass)
* `generate`       : greedy free-text continuation (explanations)
* `last_token_acts`: residual stream at the last prompt token (to build steering vectors)

Steering adds a fixed vector to the output of one transformer block at every
position, via a forward hook that stays active for the whole call it wraps.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You are a helpful assistant. Keep answers short."


class Chat:
    def __init__(self, name: str, dtype=torch.float16):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers

    # ---------------------------------------------------------------- prompts
    def render(self, conv: list[dict]) -> str:
        msgs = [{"role": "system", "content": SYSTEM}] + conv
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # templates without a system role
            msgs = [{"role": conv[0]["role"], "content": SYSTEM + "\n\n" + conv[0]["content"]}] + conv[1:]
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _enc(self, texts):
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        dev = next(self.model.parameters()).device
        return {k: v.to(dev) for k, v in enc.items()}

    def word_ids(self, word: str) -> list[int]:
        ids = set()
        for v in (word, " " + word, word.lower(), " " + word.lower()):
            e = self.tok.encode(v, add_special_tokens=False)
            if e:
                ids.add(e[0])
        return sorted(ids)

    # ------------------------------------------------------------------ reads
    @torch.no_grad()
    def choice_logits(self, convs, words: tuple[str, str], batch_size=16) -> np.ndarray:
        """log P(word0) - log P(word1) at the first answer token, per conversation."""
        a, b = self.word_ids(words[0]), self.word_ids(words[1])
        both = set(a) & set(b)
        a, b = [i for i in a if i not in both], [i for i in b if i not in both]
        out = []
        for s in range(0, len(convs), batch_size):
            enc = self._enc([self.render(c) for c in convs[s:s + batch_size]])
            lg = self.model(**enc, use_cache=False).logits[:, -1, :].float()
            out.append((torch.logsumexp(lg[:, a], -1) - torch.logsumexp(lg[:, b], -1)).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def generate(self, convs, max_new_tokens=80, batch_size=16) -> list[str]:
        res = []
        for s in range(0, len(convs), batch_size):
            enc = self._enc([self.render(c) for c in convs[s:s + batch_size]])
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            res += [t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True)]
        return res

    @torch.no_grad()
    def last_token_acts(self, convs, layer: int, batch_size=16) -> np.ndarray:
        out = []
        for s in range(0, len(convs), batch_size):
            enc = self._enc([self.render(c) for c in convs[s:s + batch_size]])
            hs = self.model(**enc, output_hidden_states=True, use_cache=False).hidden_states[layer]
            out.append(hs[:, -1, :].float().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def answer_token_acts(self, convs, word: str, layer: int, batch_size=16) -> np.ndarray:
        """Residual stream at the answer token when `word` is written as the assistant's reply."""
        out = []
        for s in range(0, len(convs), batch_size):
            texts = [self.render(c) + word for c in convs[s:s + batch_size]]
            enc = self._enc(texts)
            hs = self.model(**enc, output_hidden_states=True, use_cache=False).hidden_states[layer]
            out.append(hs[:, -1, :].float().cpu().numpy())
        return np.concatenate(out)

    # --------------------------------------------------------------- steering
    def blocks(self):
        for path in ("model.layers", "gpt_neox.layers", "transformer.h"):
            obj = self.model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise ValueError("cannot find transformer blocks")

    @contextmanager
    def steer(self, layer: int, vecs: np.ndarray | None, n_unsteered_tail: np.ndarray | None = None):
        """Add vecs[i] to example i of every batch at hidden_states[layer] (block layer-1).

        vecs: [n, d] for the batch being processed, or None for no-op.
        n_unsteered_tail: optional [n] ints. If given, the last n_unsteered_tail[i] prompt
            positions of example i are left unsteered, and so is every newly generated token.
            Used for "m2_state": the question and the model's answer are processed with
            steering on (so the cached activations of the earlier turn carry the push), while
            the follow-up question and the reply are processed without it.
        """
        if vecs is None:
            yield
            return
        block = self.blocks()[layer - 1]
        p = next(block.parameters())
        v = torch.tensor(vecs, dtype=p.dtype, device=p.device).unsqueeze(1)  # [n,1,d]

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            add = v.to(h.device)
            if add.shape[0] != h.shape[0]:  # generate() may expand; never expected with greedy
                add = add[:1]
            if n_unsteered_tail is not None:
                T = h.shape[1]
                if T == 1:  # a generation step with the KV cache: never steered
                    return output
                pos = torch.arange(T, device=h.device).view(1, T, 1)
                cut = torch.as_tensor(T - np.asarray(n_unsteered_tail), device=h.device).view(-1, 1, 1)
                add = add * (pos < cut).to(add.dtype)
            h = h + add
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

        handle = block.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


def tail_lengths(chat: Chat, convs, n_prefix_msgs: int) -> np.ndarray:
    """Number of tokens after the first n_prefix_msgs messages in each rendered conversation."""
    out = []
    for c in convs:
        full = chat.render(c)
        prefix = chat.render(c[:n_prefix_msgs])
        # render() adds a generation prompt; for the prefix we need it without one
        msgs = [{"role": "system", "content": SYSTEM}] + c[:n_prefix_msgs]
        try:
            prefix = chat.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
        if not full.startswith(prefix):  # fall back to the longest common prefix
            k = 0
            while k < min(len(full), len(prefix)) and full[k] == prefix[k]:
                k += 1
            prefix = full[:k]
        n_full = len(chat.tok(full, add_special_tokens=False).input_ids)
        n_pre = len(chat.tok(prefix, add_special_tokens=False).input_ids)
        out.append(max(1, n_full - n_pre))
    return np.asarray(out)


def batched_steered(fn, chat: Chat, convs, layer, vecs, batch_size=16, tails=None, **kw):
    """Run fn(chat, convs_batch, **kw) with a per-example steering vector."""
    outs = []
    for s in range(0, len(convs), batch_size):
        sub = convs[s:s + batch_size]
        with chat.steer(layer, None if vecs is None else vecs[s:s + batch_size],
                        None if tails is None else tails[s:s + batch_size]):
            outs.append(fn(sub, batch_size=len(sub), **kw))
    if isinstance(outs[0], np.ndarray):
        return np.concatenate(outs)
    return [x for o in outs for x in o]

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Splits one embed call into several Voyage requests.

A request may carry at most ``max_inputs`` inputs and, where Voyage documents
a ceiling, at most ``max_tokens`` tokens. Inputs are packed greedily, in order,
so concatenating the batches' results restores the input order.

Counting tokens needs the model's tokenizer, which the SDK may download on first
use, so it is avoided whenever it cannot matter. The tokenizers normalize text to
NFC and then run byte-level BPE, so every token covers at least one UTF-8 byte of
the *normalized* text. Normalization can lengthen text (U+FB2C is 3 bytes but
becomes 6, and 5 tokens), so ``len(NFC(text).encode()) + 2`` (two slack tokens
for special tokens) bounds an input's token count from above. When those bounds
already fit the token ceiling, nothing is tokenized and only the input count
splits the call.
"""

import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchLimits:
    """What one request may carry."""

    max_inputs: int
    max_tokens: int | None
    """None where Voyage documents no per-request ceiling: batch by count only."""


@dataclass(frozen=True)
class BatchItem:
    """One input, as far as batching is concerned.

    Exactly one of ``texts`` and ``fixed`` describes the cost: either the text
    pieces whose token counts add up, or a known cost that needs no counting.
    """

    texts: tuple[str, ...] = ()
    fixed: int | None = None


class OversizeInputError(ValueError):
    """One input needs more tokens than a whole request may carry."""

    def __init__(self, index: int, tokens: int, limit: int) -> None:
        """Records which input is too big, how big it is, and the limit."""
        super().__init__(f'input {index} needs {tokens} tokens; a request carries at most {limit}')
        self.index = index
        self.tokens = tokens
        self.limit = limit


TokenCounter = Callable[[list[str]], Awaitable[list[int]]]
"""Counts the tokens of each text, in order."""


def pack(costs: Sequence[int], limits: BatchLimits) -> list[range]:
    """Greedily packs inputs, in order, into requests.

    Args:
        costs: Token cost of each input.
        limits: What one request may carry.

    Returns:
        Consecutive index ranges that cover every input exactly once.

    Raises:
        OversizeInputError: An input alone costs more than ``max_tokens``, so no
            request could carry it.
    """
    batches: list[range] = []
    start = 0
    used = 0
    for index, cost in enumerate(costs):
        if limits.max_tokens is not None and cost > limits.max_tokens:
            raise OversizeInputError(index, cost, limits.max_tokens)
        full = index - start >= limits.max_inputs or (limits.max_tokens is not None and used + cost > limits.max_tokens)
        if index > start and full:
            batches.append(range(start, index))
            start = index
            used = 0
        used += cost
    if len(costs) > start:
        batches.append(range(start, len(costs)))
    return batches


def _upper_bound(item: BatchItem) -> int:
    if item.fixed is not None:
        return item.fixed
    # Normalized first: the tokenizer sees NFC text, which can be longer than the input.
    return sum(len(unicodedata.normalize('NFC', text).encode('utf-8')) + 2 for text in item.texts)


async def plan_batches(
    items: Sequence[BatchItem],
    limits: BatchLimits,
    *,
    cap: int | None,
    count_tokens: TokenCounter,
) -> list[range]:
    """Decides which inputs go into which request.

    Args:
        items: The inputs, in order.
        limits: What one request may carry.
        cap: A text's cost never exceeds this: the model's context length when
            truncation is on, because the server truncates, and None otherwise.
            It does not apply to ``fixed`` costs.
        count_tokens: Counts the tokens of each text. Only called when the
            upper bounds do not already fit the token ceiling.

    Returns:
        Consecutive index ranges that cover every input exactly once.

    Raises:
        OversizeInputError: An input alone costs more than a request may carry.
    """
    if limits.max_tokens is None:
        return pack([0] * len(items), limits)

    def capped(tokens: int) -> int:
        return tokens if cap is None else min(tokens, cap)

    bounds = [_upper_bound(item) if item.fixed is not None else capped(_upper_bound(item)) for item in items]
    if sum(bounds) <= limits.max_tokens:
        # The tokens cannot matter, so only the input count splits the call.
        return pack([0] * len(items), limits)

    pieces = [text for item in items if item.fixed is None for text in item.texts]
    counts = iter(await count_tokens(pieces) if pieces else [])
    costs: list[int] = []
    for item in items:
        if item.fixed is not None:
            costs.append(item.fixed)
        else:
            costs.append(capped(sum(next(counts) for _ in item.texts)))
    return pack(costs, limits)

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

"""Splitting a call into requests: the packing rules, then the embedders that use them."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from genkit_voyageai._batching import BatchItem, BatchLimits, OversizeInputError, pack, plan_batches
from genkit_voyageai._catalog import CATALOG
from genkit_voyageai._embedders import VoyageEmbedder
from voyageai.object.contextualized_embeddings import ContextualizedEmbeddingsResult

from genkit import Document, GenkitError, Part
from genkit.embedder import EmbedRequest

# Packing -------------------------------------------------------------------


def test_pack_fills_each_request_in_order_up_to_either_limit() -> None:
    assert pack([1] * 5, BatchLimits(max_inputs=2, max_tokens=100)) == [range(0, 2), range(2, 4), range(4, 5)]
    assert pack([6, 5, 4, 9], BatchLimits(max_inputs=10, max_tokens=10)) == [range(0, 1), range(1, 3), range(3, 4)]


def test_pack_accepts_an_exact_fit_and_splits_one_token_over() -> None:
    assert pack([4, 6], BatchLimits(max_inputs=10, max_tokens=10)) == [range(0, 2)]
    assert pack([4, 7], BatchLimits(max_inputs=10, max_tokens=10)) == [range(0, 1), range(1, 2)]


def test_pack_without_a_token_ceiling_splits_by_count_only() -> None:
    assert pack([10**9] * 3, BatchLimits(max_inputs=2, max_tokens=None)) == [range(0, 2), range(2, 3)]


def test_pack_names_the_input_that_fits_no_request() -> None:
    with pytest.raises(OversizeInputError) as raised:
        pack([3, 11, 2], BatchLimits(max_inputs=10, max_tokens=10))
    assert (raised.value.index, raised.value.tokens, raised.value.limit) == (1, 11, 10)


# Planning ------------------------------------------------------------------


class _Counter:
    """A tokenizer double: one token per character, and a record of what it was asked."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def __call__(self, texts: list[str]) -> list[int]:
        self.calls.append(texts)
        return [len(t) for t in texts]


@pytest.mark.asyncio
async def test_plan_never_tokenizes_when_the_upper_bounds_fit() -> None:
    counter = _Counter()
    items = [BatchItem(texts=('hello',)) for _ in range(5)]
    # 5 bytes + 2 specials each, 35 in all.
    assert await plan_batches(items, BatchLimits(10, 35), cap=None, count_tokens=counter) == [range(0, 5)]
    assert await plan_batches(items, BatchLimits(2, 35), cap=None, count_tokens=counter) == [
        range(0, 2),
        range(2, 4),
        range(4, 5),
    ]
    assert counter.calls == []


@pytest.mark.asyncio
async def test_plan_bounds_the_normalized_text_not_the_raw_bytes() -> None:
    counter = _Counter()
    # U+FB2C is 3 bytes, but NFC turns it into 6 (and the tokenizer sees 5 tokens each).
    text = '\ufb2c' * 100
    assert len(text.encode()) + 2 == 302
    items = [BatchItem(texts=(text,))]
    # 302 raw-byte tokens would fit 400; the normalized bound of 602 does not, so it must count.
    assert await plan_batches(items, BatchLimits(10, 400), cap=None, count_tokens=counter) == [range(0, 1)]
    assert counter.calls == [[text]]


@pytest.mark.asyncio
async def test_plan_without_a_ceiling_never_tokenizes() -> None:
    counter = _Counter()
    items = [BatchItem(texts=('x' * 10_000,)) for _ in range(3)]
    assert await plan_batches(items, BatchLimits(2, None), cap=None, count_tokens=counter) == [
        range(0, 2),
        range(2, 3),
    ]
    assert counter.calls == []


@pytest.mark.asyncio
async def test_plan_counts_exactly_once_the_bounds_do_not_fit() -> None:
    counter = _Counter()
    items = [BatchItem(texts=('aaaa',)), BatchItem(fixed=7), BatchItem(texts=('bb', 'cc'))]
    # Bounds 6 + 7 + 8 = 21 > 12, so it counts: 4 + 7 + (2 + 2).
    assert await plan_batches(items, BatchLimits(10, 12), cap=None, count_tokens=counter) == [
        range(0, 2),
        range(2, 3),
    ]
    assert counter.calls == [['aaaa', 'bb', 'cc']]


@pytest.mark.asyncio
async def test_plan_caps_a_text_at_the_context_length_only_when_truncating() -> None:
    items = [BatchItem(texts=('x' * 50,))]
    limits = BatchLimits(10, 20)
    assert await plan_batches(items, limits, cap=20, count_tokens=_Counter()) == [range(0, 1)]
    with pytest.raises(OversizeInputError):
        await plan_batches(items, limits, cap=None, count_tokens=_Counter())


# Embedders -----------------------------------------------------------------


def _client() -> MagicMock:
    client = MagicMock()
    client.tokenize = MagicMock(side_effect=AssertionError('tokenizer must not be used'))
    return client


def _request(*documents: Document | str, **options: Any) -> EmbedRequest[Any]:
    docs = [Document.from_text(d) if isinstance(d, str) else d for d in documents]
    return EmbedRequest(input=docs, options=options or None)


def _text_vectors(*, texts: list[str], **_: Any) -> MagicMock:
    # The vector carries the text's number, so order across requests is visible.
    return MagicMock(embeddings=[[float(t.removeprefix('d'))] for t in texts])


@pytest.mark.asyncio
async def test_text_calls_over_1000_inputs_split_and_come_back_in_order() -> None:
    client = _client()
    client.embed = AsyncMock(side_effect=_text_vectors)
    response = await VoyageEmbedder(CATALOG['voyage-4'], client).embed(_request(*[f'd{i}' for i in range(2001)]))
    assert [len(call.kwargs['texts']) for call in client.embed.call_args_list] == [1000, 1000, 1]
    assert [e.embedding[0] for e in response.embeddings] == [float(i) for i in range(2001)]


@pytest.mark.asyncio
async def test_code_4_has_no_token_ceiling_so_only_the_count_splits_it() -> None:
    client = _client()
    client.embed = AsyncMock(side_effect=lambda *, texts, **_: MagicMock(embeddings=[[1.0]] * len(texts)))
    huge = 'x' * 500_000
    await VoyageEmbedder(CATALOG['voyage-code-4'], client).embed(_request(huge, huge, huge, huge))
    assert client.embed.await_count == 1
    client.tokenize.assert_not_called()


@pytest.mark.asyncio
async def test_exact_counts_decide_the_split_once_the_bounds_do_not_fit() -> None:
    client = _client()
    client.tokenize = MagicMock(return_value=[[0] * 70_000, [0] * 60_000, [0] * 20_000])
    client.embed = AsyncMock(side_effect=lambda *, texts, **_: MagicMock(embeddings=[[1.0]] * len(texts)))
    texts = ['a' * 200_000, 'b' * 100_000, 'c' * 50_000]
    # voyage-3-large carries 120000 tokens per request; truncation is off so nothing is capped.
    await VoyageEmbedder(CATALOG['voyage-3-large'], client).embed(_request(*texts, truncation=False))
    assert [len(call.kwargs['texts']) for call in client.embed.call_args_list] == [1, 2]
    assert client.tokenize.call_args.args == (texts, 'voyage-3-large')


@pytest.mark.asyncio
async def test_a_tokenizer_failure_fails_closed_and_names_the_cause() -> None:
    client = _client()
    client.tokenize = MagicMock(side_effect=OSError('offline'))
    client.embed = AsyncMock()
    with pytest.raises(GenkitError, match='tokenizer failed to load.*OSError.*offline') as raised:
        await VoyageEmbedder(CATALOG['voyage-3-large'], client).embed(
            _request('a' * 300_000, 'b' * 300_000, truncation=False)
        )
    assert raised.value.status == 'UNAVAILABLE'
    client.embed.assert_not_called()


@pytest.mark.asyncio
async def test_a_contextualized_document_over_the_request_ceiling_is_rejected_by_index() -> None:
    client = _client()
    client.tokenize = MagicMock(return_value=[[0] * 10, [0] * 130_000])
    client.contextualized_embed = AsyncMock()
    with pytest.raises(GenkitError, match='document 1 needs 130000 tokens.*at most 120000') as raised:
        await VoyageEmbedder(CATALOG['voyage-context-4'], client).embed(
            _request('a' * 100, 'b' * 400_000, input_type='document')
        )
    assert raised.value.status == 'INVALID_ARGUMENT'
    client.contextualized_embed.assert_not_called()


@pytest.mark.asyncio
async def test_contextualized_document_index_stays_global_across_requests() -> None:
    client = _client()
    results = iter([
        MagicMock(results=[ContextualizedEmbeddingsResult(index=0, embeddings=[[1.0], [2.0]], chunk_texts=['a', 'b'])]),
        MagicMock(results=[ContextualizedEmbeddingsResult(index=0, embeddings=[[3.0]], chunk_texts=['c'])]),
    ])
    client.contextualized_embed = AsyncMock(side_effect=lambda **_: next(results))
    embedder = VoyageEmbedder(CATALOG['voyage-context-4'], client, limits=BatchLimits(max_inputs=1, max_tokens=120_000))
    response = await embedder.embed(_request('first', 'second', input_type='document'))
    assert [(e.embedding[0], e.metadata) for e in response.embeddings] == [
        (1.0, {'documentIndex': 0, 'chunkIndex': 0, 'text': 'a'}),
        (2.0, {'documentIndex': 0, 'chunkIndex': 1, 'text': 'b'}),
        (3.0, {'documentIndex': 1, 'chunkIndex': 0, 'text': 'c'}),
    ]


@pytest.mark.asyncio
async def test_multimodal_media_costs_the_per_input_maximum_without_tokenizing() -> None:
    client = _client()
    client.multimodal_embed = AsyncMock(side_effect=lambda *, inputs, **_: MagicMock(embeddings=[[1.0]] * len(inputs)))
    image = Document(content=[Part.from_text('hi'), Part.from_media('data:image/png;base64,iVBORw0KGgo=', 'image/png')])
    embedder = VoyageEmbedder(CATALOG['voyage-multimodal-3.5'], client, limits=BatchLimits(1000, 50_000))
    await embedder.embed(_request(image, image, image))
    # 32000 tokens each against 50000: one image per request.
    assert [len(call.kwargs['inputs']) for call in client.multimodal_embed.call_args_list] == [1, 1, 1]
    client.tokenize.assert_not_called()

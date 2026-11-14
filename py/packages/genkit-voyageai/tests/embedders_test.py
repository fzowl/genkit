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

"""Failure modes of the Voyage embedders, against a stubbed SDK client."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from genkit_voyageai._catalog import CATALOG
from genkit_voyageai._embedders import VoyageEmbedder
from voyageai.object.contextualized_embeddings import ContextualizedEmbeddingsResult

from genkit import Document, GenkitError, Part
from genkit.embedder import EmbedRequest

PNG_DATA_URL = 'data:image/png;base64,iVBORw0KGgo='
MP4_DATA_URL = 'data:video/mp4;base64,AAAAIGZ0eXA='


def _client(**results: Any) -> MagicMock:
    client = MagicMock()
    client.embed = AsyncMock(return_value=results.get('embed'))
    client.contextualized_embed = AsyncMock(return_value=results.get('contextualized'))
    client.multimodal_embed = AsyncMock(return_value=results.get('multimodal'))
    # Unit tests never reach the tokenizer unless they set it up on purpose.
    client.tokenize = MagicMock(side_effect=AssertionError('tokenizer must not be used'))
    return client


def _request(*documents: Document | str, **options: Any) -> EmbedRequest[Any]:
    docs = [Document.from_text(d) if isinstance(d, str) else d for d in documents]
    return EmbedRequest(input=docs, options=options or None)


def _embedder(model: str, client: MagicMock) -> VoyageEmbedder:
    return VoyageEmbedder(CATALOG[model], client)


def _media(url: str, content_type: str | None = None) -> Document:
    return Document(content=[Part.from_media(url, content_type)])


async def _rejected(model: str, request: EmbedRequest[Any], match: str) -> MagicMock:
    client = _client()
    with pytest.raises(GenkitError, match=match) as raised:
        await _embedder(model, client).embed(request)
    assert raised.value.status == 'INVALID_ARGUMENT'
    # Rejected locally: nothing was sent.
    for call in (client.embed, client.contextualized_embed, client.multimodal_embed):
        call.assert_not_called()
    return client


# Text ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('model', ['voyage-4', 'voyage-context-4', 'voyage-multimodal-3.5'])
async def test_empty_input_returns_an_empty_response_without_a_request(model: str) -> None:
    client = _client()
    response = await _embedder(model, client).embed(EmbedRequest(input=[]))
    assert response.embeddings == []
    for call in (client.embed, client.contextualized_embed, client.multimodal_embed):
        call.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model', 'dimension', 'match'),
    [
        ('voyage-4', 768, 'use one of 256, 512, 1024, 2048'),
        ('voyage-finance-2', 256, 'only produces 1024-dimensional'),
        ('voyage-law-2', 2048, 'only produces 1024-dimensional'),
        ('voyage-multimodal-3', 512, 'only produces 1024-dimensional'),
        ('voyage-context-4', 128, 'use one of'),
    ],
)
async def test_output_dimension_outside_the_model_dimensions_is_rejected(
    model: str, dimension: int, match: str
) -> None:
    document = Document.from_text('x')
    await _rejected(model, _request(document, input_type='document', output_dimension=dimension), match)


@pytest.mark.asyncio
async def test_text_rejects_media_and_empty_documents() -> None:
    await _rejected('voyage-4', _request(_media(PNG_DATA_URL)), 'document 0 has a non-text part')
    await _rejected('voyage-4', _request('fine', '   '), 'document 1 has no text')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'options', [{'input_type': 'passage'}, {'inputType': 'query', 'input_type': 'query'}, {'dim': 1}]
)
async def test_options_that_are_not_part_of_the_config_are_rejected(options: dict[str, Any]) -> None:
    await _rejected('voyage-4', _request('x', **options), 'invalid options')


# Contextualized -------------------------------------------------------------


@pytest.mark.asyncio
async def test_contextualized_requires_an_input_type() -> None:
    await _rejected('voyage-context-4', _request('x'), 'input_type is required')


@pytest.mark.asyncio
async def test_contextualized_rejects_pre_chunked_and_non_text_documents() -> None:
    chunked = Document(content=[Part.from_text('one'), Part.from_text('two')])
    await _rejected(
        'voyage-context-4', _request(chunked, input_type='document'), 'pre-chunked documents are not supported'
    )
    await _rejected('voyage-context-4', _request(chunked, input_type='query'), 'each query must be a single string')
    await _rejected('voyage-context-4', _request(_media(PNG_DATA_URL), input_type='document'), 'non-text part')


@pytest.mark.asyncio
async def test_contextualized_document_mode_auto_chunks_and_query_mode_never_does() -> None:
    result = MagicMock(results=[ContextualizedEmbeddingsResult(index=0, embeddings=[[0.5]], chunk_texts=None)])
    client = _client(contextualized=result)

    await _embedder('voyage-context-4', client).embed(_request('doc', input_type='document'))
    document_call = client.contextualized_embed.call_args.kwargs
    assert document_call['inputs'] == ['doc']
    assert document_call['enable_auto_chunking'] is True
    assert document_call['chunk_size'] == 32_000
    assert 'chunk_overlap' not in document_call

    await _embedder('voyage-context-4', client).embed(_request('query', input_type='query'))
    query_call = client.contextualized_embed.call_args.kwargs
    assert query_call['enable_auto_chunking'] is False
    assert query_call['chunk_size'] is None


@pytest.mark.asyncio
async def test_contextualized_emits_one_embedding_per_chunk_with_text_only_when_returned() -> None:
    result = MagicMock(
        results=[
            ContextualizedEmbeddingsResult(index=0, embeddings=[[1.0], [2.0]], chunk_texts=['a', 'b']),
            ContextualizedEmbeddingsResult(index=1, embeddings=[[3.0]], chunk_texts=None),
        ]
    )
    response = await _embedder('voyage-context-4', _client(contextualized=result)).embed(
        _request('doc one', 'doc two', input_type='document')
    )
    assert [(e.embedding, e.metadata) for e in response.embeddings] == [
        ([1.0], {'documentIndex': 0, 'chunkIndex': 0, 'text': 'a'}),
        ([2.0], {'documentIndex': 0, 'chunkIndex': 1, 'text': 'b'}),
        ([3.0], {'documentIndex': 1, 'chunkIndex': 0}),
    ]


@pytest.mark.asyncio
async def test_a_response_that_does_not_match_the_input_is_an_internal_error() -> None:
    result = MagicMock(results=[ContextualizedEmbeddingsResult(index=0, embeddings=[[1.0]], chunk_texts=None)])
    with pytest.raises(GenkitError, match='returned 1 results for 2 inputs') as raised:
        await _embedder('voyage-context-4', _client(contextualized=result)).embed(
            _request('a', 'b', input_type='document')
        )
    assert raised.value.status == 'INTERNAL'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model', 'call', 'result', 'options'),
    [
        ('voyage-4', 'embed', MagicMock(embeddings=[[]]), {}),
        ('voyage-multimodal-3.5', 'multimodal_embed', MagicMock(embeddings=[[]]), {}),
        (
            'voyage-context-4',
            'contextualized_embed',
            MagicMock(results=[ContextualizedEmbeddingsResult(index=0, embeddings=[], chunk_texts=None)]),
            {'input_type': 'document'},
        ),
        (
            'voyage-context-4',
            'contextualized_embed',
            MagicMock(
                results=[ContextualizedEmbeddingsResult(index=0, embeddings=[[1.0], []], chunk_texts=['a', 'b'])]
            ),
            {'input_type': 'document'},
        ),
    ],
)
async def test_a_response_with_an_empty_vector_is_an_internal_error(
    model: str, call: str, result: MagicMock, options: dict[str, Any]
) -> None:
    client = _client()
    getattr(client, call).return_value = result
    with pytest.raises(GenkitError, match='empty embedding|no embeddings') as raised:
        await _embedder(model, client).embed(_request('x', **options))
    assert raised.value.status == 'INTERNAL'


# Multimodal -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_multimodal_rejects_mixing_url_and_base64_media_in_one_request() -> None:
    request = _request(_media('https://example.com/a.png'), _media(PNG_DATA_URL))
    await _rejected('voyage-multimodal-3.5', request, 'either URLs or base64 data URLs')
    await _rejected(
        'voyage-multimodal-3.5', _request(_media(PNG_DATA_URL), _media('https://example.com/a.png')), 'earlier input'
    )


@pytest.mark.asyncio
async def test_multimodal_rejects_unsupported_media() -> None:
    await _rejected(
        'voyage-multimodal-3.5', _request(_media('data:image/bmp;base64,Qk0=')), "'image/bmp' is not supported"
    )
    await _rejected(
        'voyage-multimodal-3.5', _request(_media('https://example.com/clip.mov')), "'video/quicktime' is not supported"
    )
    await _rejected('voyage-multimodal-3', _request(_media(MP4_DATA_URL)), 'does not accept video')
    await _rejected('voyage-multimodal-3.5', _request(_media('https://example.com/blob')), 'cannot tell the media type')
    await _rejected(
        'voyage-multimodal-3.5',
        _request(_media('ftp://example.com/a.png', 'image/png')),
        r'http\(s\) URL or a base64 data URL',
    )
    await _rejected('voyage-multimodal-3.5', _request(_media('data:image/png,rawbytes')), '<type>;base64,<data>')
    await _rejected(
        'voyage-multimodal-3.5', _request(_media(PNG_DATA_URL, 'image/jpeg')), 'does not match the data URL type'
    )


@pytest.mark.asyncio
async def test_multimodal_rejects_a_document_without_content() -> None:
    await _rejected(
        'voyage-multimodal-3.5', _request(Document(content=[Part.from_text('')])), 'document 0 has no content'
    )


@pytest.mark.asyncio
async def test_multimodal_sends_parts_in_order_and_the_requested_dimension() -> None:
    result = MagicMock(embeddings=[[0.25]])
    client = _client(multimodal=result)
    document = Document(content=[Part.from_text('a cat'), Part.from_media(PNG_DATA_URL, 'image/png')])
    await _embedder('voyage-multimodal-3.5', client).embed(_request(document, output_dimension=256))
    call = client.multimodal_embed.call_args.kwargs
    assert call['inputs'] == [
        {'content': [{'type': 'text', 'text': 'a cat'}, {'type': 'image_base64', 'image_base64': PNG_DATA_URL}]}
    ]
    assert call['output_dimension'] == 256

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

"""Live Voyage AI calls; skipped unless VOYAGE_API_KEY is set. Each test sends a few tokens."""

import base64
import io
import os

import pytest
from genkit_voyageai import VoyageAI

from genkit import Document, Genkit, GenkitError, Part

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not os.environ.get('VOYAGE_API_KEY'), reason='VOYAGE_API_KEY not set'),
]


def _png_data_url() -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new('RGB', (32, 32), (200, 40, 40)).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


async def test_text_embedder_honours_dimension_and_input_type() -> None:
    ai = Genkit(plugins=[VoyageAI()])
    ref = VoyageAI.embedder('voyage-4-lite')
    document = await ai.embed(
        embedder=ref, content='Smoked salmon tartine', config={'input_type': 'document', 'output_dimension': 256}
    )
    query = await ai.embed(embedder=ref, content='fish lunch', config={'input_type': 'query', 'output_dimension': 256})
    assert len(document[0].embedding) == len(query[0].embedding) == 256


async def test_contextualized_document_comes_back_chunked_with_text() -> None:
    ai = Genkit(plugins=[VoyageAI()])
    ref = VoyageAI.embedder('voyage-context-4')
    chunks = await ai.embed(
        embedder=ref,
        content='Smoked salmon tartine. It is served on rye.',
        config={'input_type': 'document', 'output_dimension': 256},
    )
    assert chunks and all(len(c.embedding) == 256 for c in chunks)
    assert all(isinstance(c.metadata and c.metadata.get('text'), str) for c in chunks)
    assert [c.metadata['chunkIndex'] for c in chunks if c.metadata] == list(range(len(chunks)))

    queries = await ai.embed_many(
        embedder=ref, content=['fish lunch'], config={'input_type': 'query', 'output_dimension': 256}
    )
    assert len(queries) == 1 and queries[0].metadata == {'documentIndex': 0, 'chunkIndex': 0}


async def test_multimodal_embeds_text_with_a_base64_image() -> None:
    ai = Genkit(plugins=[VoyageAI()])
    document = Document(content=[Part.from_text('a red square'), Part.from_media(_png_data_url(), 'image/png')])
    vectors = await ai.embed(
        embedder=VoyageAI.embedder('voyage-multimodal-3.5'),
        content=document,
        config={'input_type': 'document', 'output_dimension': 256},
    )
    assert len(vectors) == 1 and len(vectors[0].embedding) == 256


async def test_a_rejected_key_reports_voyages_detail() -> None:
    ai = Genkit(plugins=[VoyageAI(api_key='pa-not-a-real-key')])
    with pytest.raises(GenkitError) as raised:
        await ai.embed(embedder=VoyageAI.embedder('voyage-4-lite'), content='x')
    assert raised.value.status == 'UNAUTHENTICATED'
    assert 'invalid' in raised.value.message.lower()

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

"""Voyage AI embedder catalog.

There is deliberately no default model: every call names one, and a name that
is not listed here does not resolve.
"""

from dataclasses import dataclass
from typing import Literal

EmbedderKind = Literal['text', 'contextualized', 'multimodal']
"""Which Voyage endpoint serves a model: embeddings, contextualized embeddings, or multimodal embeddings."""

MAX_INPUTS_PER_REQUEST = 1000
"""Inputs one request may carry, on all three endpoints. Larger calls are split into several requests."""

CONTEXTUALIZED_CHUNK_SIZE = 32_000
"""Chunk size, in tokens, sent with every auto-chunked contextualized request."""


@dataclass(frozen=True)
class VoyageModel:
    """One catalog entry.

    ``context_tokens`` and ``max_request_tokens`` are documented metadata: the
    server enforces them (a 400 carrying a ``detail``), this plugin does not.
    ``max_request_tokens`` is None where Voyage does not document a ceiling.
    """

    name: str
    kind: EmbedderKind
    dimensions: tuple[int, ...]
    """Supported ``output_dimension`` values; the first is the model's default."""
    context_tokens: int
    max_request_tokens: int | None
    inputs: tuple[str, ...] = ('text',)
    """Genkit input kinds the model accepts."""

    @property
    def default_dimension(self) -> int:
        """The width of a vector when no ``output_dimension`` is requested."""
        return self.dimensions[0]


_FLEXIBLE = (1024, 2048, 512, 256)
_FIXED = (1024,)


def _text(
    name: str, *, max_request_tokens: int | None, dimensions: tuple[int, ...] = _FLEXIBLE, context_tokens: int = 32_000
) -> VoyageModel:
    return VoyageModel(
        name=name,
        kind='text',
        dimensions=dimensions,
        context_tokens=context_tokens,
        max_request_tokens=max_request_tokens,
    )


_ENTRIES: tuple[VoyageModel, ...] = (
    _text('voyage-4-large', max_request_tokens=120_000),
    _text('voyage-4', max_request_tokens=320_000),
    _text('voyage-4-lite', max_request_tokens=1_000_000),
    # Voyage documents no per-request ceiling for voyage-code-4.
    _text('voyage-code-4', max_request_tokens=None),
    _text('voyage-3-large', max_request_tokens=120_000),
    _text('voyage-3.5', max_request_tokens=320_000),
    _text('voyage-3.5-lite', max_request_tokens=1_000_000),
    _text('voyage-code-3', max_request_tokens=120_000),
    _text('voyage-finance-2', max_request_tokens=120_000, dimensions=_FIXED),
    _text('voyage-law-2', max_request_tokens=120_000, dimensions=_FIXED, context_tokens=16_000),
    VoyageModel(
        name='voyage-context-4',
        kind='contextualized',
        dimensions=(1024, 2048, 512, 256),
        context_tokens=32_000,
        max_request_tokens=120_000,
    ),
    VoyageModel(
        name='voyage-context-3',
        kind='contextualized',
        dimensions=(1024, 2048, 512, 256),
        context_tokens=32_000,
        max_request_tokens=120_000,
    ),
    VoyageModel(
        name='voyage-multimodal-3.5',
        kind='multimodal',
        dimensions=(1024, 256, 512, 2048),
        context_tokens=32_000,
        max_request_tokens=320_000,
        inputs=('text', 'image', 'video'),
    ),
    VoyageModel(
        name='voyage-multimodal-3',
        kind='multimodal',
        dimensions=_FIXED,
        context_tokens=32_000,
        max_request_tokens=320_000,
        inputs=('text', 'image'),
    ),
)

CATALOG: dict[str, VoyageModel] = {entry.name: entry for entry in _ENTRIES}
"""Every embedder this plugin serves, keyed by Voyage model name."""

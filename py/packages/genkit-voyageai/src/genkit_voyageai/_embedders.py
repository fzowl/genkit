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

"""Voyage AI embedders: text, contextualized and multimodal.

All three go through the official ``voyageai`` async client. The three
endpoints differ in what a Genkit ``Document`` may hold and in what comes back:

- text: one text-only document, one vector.
- contextualized: one text document, one vector per chunk. In 'document' mode
  Voyage splits each document itself (``enable_auto_chunking``), so a single
  document can yield several vectors; each carries its ``documentIndex``,
  ``chunkIndex`` and, when Voyage returned it, the chunk ``text`` in metadata.
- multimodal: text, image and video parts of one document become one input and
  one vector.
"""

import asyncio
import mimetypes
from collections.abc import Sequence
from typing import Any, TypeVar
from urllib.parse import urlsplit

from pydantic import ValidationError
from voyageai import error as voyage_error
from voyageai.client_async import AsyncClient

from genkit import Document, Embedding, ErrorReason, GenkitError, Part
from genkit.embedder import EmbedRequest, EmbedResponse
from genkit_voyageai._batching import BatchItem, BatchLimits, OversizeInputError, plan_batches
from genkit_voyageai._catalog import (
    CONTEXTUALIZED_CHUNK_SIZE,
    MAX_INPUTS_PER_REQUEST,
    VoyageModel,
)
from genkit_voyageai._config import VoyageContextualizedEmbedConfig, VoyageEmbedConfig
from genkit_voyageai._errors import to_genkit_error

_ConfigT = TypeVar('_ConfigT', VoyageEmbedConfig, VoyageContextualizedEmbedConfig)

IMAGE_MEDIA_TYPES = frozenset({'image/png', 'image/jpeg', 'image/webp', 'image/gif'})
VIDEO_MEDIA_TYPES = frozenset({'video/mp4'})


class VoyageEmbedder:
    """Runs one embed request against one Voyage model."""

    def __init__(self, model: VoyageModel, client: AsyncClient, *, limits: BatchLimits | None = None) -> None:
        """Initializes the embedder.

        Args:
            model: The catalog entry; it decides which endpoint is called.
            client: The Voyage SDK async client, already holding the API key.
            limits: What one request may carry, to override the catalog's
                (1000 inputs and the model's documented token ceiling).
        """
        self._model = model
        self._client = client
        self._limits = limits or BatchLimits(MAX_INPUTS_PER_REQUEST, model.max_request_tokens)

    async def embed(self, request: EmbedRequest[Any]) -> EmbedResponse:
        """Embeds the request's documents.

        Everything that can be checked locally is checked before the first
        request goes out, so a bad batch costs nothing.

        Args:
            request: The Genkit embed request; ``options`` carries the config.

        A call larger than one request may carry (1000 inputs, or the model's
        documented token ceiling) is split into several requests, sent one
        after another, and the results are joined in input order.

        Returns:
            An empty response, with no request sent, when there are no
            documents. Otherwise one embedding per document, or for contextualized embedders in
            'document' mode one per chunk, flattened in (document, chunk) order.

        Raises:
            GenkitError: INVALID_ARGUMENT for input or options Voyage would
                reject, including one input too big for any request; UNAVAILABLE
                when the tokenizer needed to split a large call cannot be
                loaded; INTERNAL for a response that cannot be matched to the
                input or carries an empty vector, and the mapped status (carrying Voyage's ``detail``)
                for a failed call.
        """
        documents = request.input
        if not documents:
            return EmbedResponse(embeddings=[])
        if self._model.kind == 'contextualized':
            return await self._embed_contextualized(documents, self._parse(VoyageContextualizedEmbedConfig, request))
        options = self._parse(VoyageEmbedConfig, request)
        if self._model.kind == 'multimodal':
            return await self._embed_multimodal(documents, options)
        return await self._embed_text(documents, options)

    # Text -----------------------------------------------------------------

    async def _embed_text(self, documents: list[Document], options: VoyageEmbedConfig) -> EmbedResponse:
        self._check_dimension(options.output_dimension)
        texts = [self._text_only(document, index) for index, document in enumerate(documents)]
        truncation = True if options.truncation is None else options.truncation
        batches = await self._plan([BatchItem(texts=(text,)) for text in texts], cap=self._cap(truncation))
        embeddings: list[Embedding] = []
        for batch in batches:
            chunk = texts[batch.start : batch.stop]
            try:
                result = await self._client.embed(
                    texts=chunk,
                    model=self._model.name,
                    input_type=options.input_type,
                    truncation=truncation,
                    output_dimension=options.output_dimension,
                )
            except voyage_error.VoyageError as error:
                raise to_genkit_error(error, model=self._model.name) from error
            self._check_returned(len(result.embeddings), len(chunk))
            embeddings.extend(Embedding(embedding=self._vector(vec)) for vec in result.embeddings)
        return EmbedResponse(embeddings=embeddings)

    def _text_only(self, document: Document, index: int) -> str:
        for part in document.content:
            if part.text is None:
                raise self._invalid(
                    f'document {index} has a non-text part; {self._model.name} embeds text only '
                    f'(use a voyage-multimodal-* model for images and video)'
                )
        text = document.text
        if not text.strip():
            raise self._invalid(f'document {index} has no text')
        return text

    # Contextualized -------------------------------------------------------

    async def _embed_contextualized(
        self, documents: list[Document], options: VoyageContextualizedEmbedConfig
    ) -> EmbedResponse:
        if options.input_type is None:
            raise self._invalid("input_type is required: set it to 'document' or 'query'")
        self._check_dimension(options.output_dimension)
        document_mode = options.input_type == 'document'
        texts = [self._single_text(document, index, document_mode) for index, document in enumerate(documents)]
        # Auto-chunked documents are not cut to the context length, so no cap:
        # a document costs all its tokens against the request ceiling.
        batches = await self._plan([BatchItem(texts=(text,)) for text in texts], cap=None)
        embeddings: list[Embedding] = []
        for batch in batches:
            chunk = texts[batch.start : batch.stop]
            try:
                # Auto-chunking only works for 'document' and then wants one plain
                # string per document; queries are never chunked.
                result = await self._client.contextualized_embed(
                    inputs=chunk,
                    model=self._model.name,
                    input_type=options.input_type,
                    output_dimension=options.output_dimension,
                    enable_auto_chunking=document_mode,
                    chunk_size=CONTEXTUALIZED_CHUNK_SIZE if document_mode else None,
                )
            except voyage_error.VoyageError as error:
                raise to_genkit_error(error, model=self._model.name) from error
            self._check_returned(len(result.results), len(chunk))
            for local_index, item in enumerate(result.results):
                # documentIndex counts across the whole call, not within this request.
                document_index = batch.start + local_index
                chunk_texts = item.chunk_texts
                if chunk_texts is not None and len(chunk_texts) != len(item.embeddings):
                    raise GenkitError(
                        status='INTERNAL',
                        message=(
                            f'voyageai {self._model.name}: document {document_index} came back with '
                            f'{len(item.embeddings)} embeddings and {len(chunk_texts)} chunk texts'
                        ),
                    )
                if not item.embeddings:
                    raise GenkitError(
                        status='INTERNAL',
                        message=f'voyageai {self._model.name}: document {document_index} came back with no embeddings',
                    )
                for chunk_index, vector in enumerate(item.embeddings):
                    metadata: dict[str, Any] = {'documentIndex': document_index, 'chunkIndex': chunk_index}
                    if chunk_texts is not None:
                        metadata['text'] = chunk_texts[chunk_index]
                    embeddings.append(Embedding(embedding=self._vector(vector), metadata=metadata))
        return EmbedResponse(embeddings=embeddings)

    def _single_text(self, document: Document, index: int, document_mode: bool) -> str:
        parts = document.content
        if len(parts) != 1 or parts[0].text is None:
            if any(part.text is None for part in parts):
                raise self._invalid(
                    f'document {index} has a non-text part; {self._model.name} embeds text only '
                    f'(use a voyage-multimodal-* model for images and video)'
                )
            what = (
                'pre-chunked documents are not supported, pass the whole document and let Voyage chunk it'
                if document_mode
                else 'each query must be a single string'
            )
            raise self._invalid(f'document {index} has {len(parts)} parts, expected exactly one text part: {what}')
        text = parts[0].text
        if not text.strip():
            raise self._invalid(f'document {index} has no text')
        return text

    # Multimodal -----------------------------------------------------------

    async def _embed_multimodal(self, documents: list[Document], options: VoyageEmbedConfig) -> EmbedResponse:
        self._check_dimension(options.output_dimension)
        # Which form media took in the request so far: 'url' or 'base64'.
        encodings: list[str] = []
        inputs = [{'content': self._segments(document, index, encodings)} for index, document in enumerate(documents)]
        truncation = True if options.truncation is None else options.truncation
        # Text tokens say nothing about images and video, which Voyage prices by
        # pixels (unknowable for a URL). To stay sound rather than optimistic, an
        # input with any media costs the documented per-input maximum, the model's
        # context length; text-only inputs are counted like text.
        items = [
            BatchItem(fixed=self._model.context_tokens)
            if any(segment['type'] != 'text' for segment in item['content'])
            else BatchItem(texts=tuple(segment['text'] for segment in item['content']))
            for item in inputs
        ]
        batches = await self._plan(items, cap=self._cap(truncation))
        embeddings: list[Embedding] = []
        for batch in batches:
            chunk = inputs[batch.start : batch.stop]
            try:
                result = await self._client.multimodal_embed(
                    inputs=chunk,
                    model=self._model.name,
                    input_type=options.input_type,
                    truncation=truncation,
                    output_dimension=options.output_dimension,
                )
            except voyage_error.VoyageError as error:
                raise to_genkit_error(error, model=self._model.name) from error
            self._check_returned(len(result.embeddings), len(chunk))
            embeddings.extend(Embedding(embedding=self._vector(vec)) for vec in result.embeddings)
        return EmbedResponse(embeddings=embeddings)

    def _segments(self, document: Document, index: int, encodings: list[str]) -> list[dict[str, str]]:
        segments: list[dict[str, str]] = []
        for part in document.content:
            if part.media is not None:
                segments.append(self._media_segment(part, index, encodings))
            elif part.text is not None:
                if part.text:
                    segments.append({'type': 'text', 'text': part.text})
            else:
                raise self._invalid(f'document {index} has a part that is neither text nor media')
        if not segments:
            raise self._invalid(f'document {index} has no content')
        return segments

    def _media_segment(self, part: Part, index: int, encodings: list[str]) -> dict[str, str]:
        media = part.media
        assert media is not None  # noqa: S101 - narrowed by the caller
        url = media.url
        declared = (media.content_type or '').split(';', 1)[0].strip().lower()
        if url.startswith('data:'):
            encoding = 'base64'
            header, _, payload = url.partition(',')
            fields = header.removeprefix('data:').split(';')
            media_type = fields[0].strip().lower()
            if 'base64' not in fields[1:] or not media_type or not payload:
                raise self._invalid(f"document {index}: a media data URL must look like 'data:<type>;base64,<data>'")
            if declared and declared != media_type:
                raise self._invalid(
                    f'document {index}: content type {declared!r} does not match the data URL type {media_type!r}'
                )
        elif url.startswith(('http://', 'https://')):
            encoding = 'url'
            media_type = declared or (mimetypes.guess_type(urlsplit(url).path)[0] or '')
            if not media_type:
                raise self._invalid(
                    f'document {index}: cannot tell the media type of {url!r}; set the part content type'
                )
        else:
            raise self._invalid(f'document {index}: media must be an http(s) URL or a base64 data URL')

        if media_type in IMAGE_MEDIA_TYPES:
            kind = 'image'
        elif media_type in VIDEO_MEDIA_TYPES:
            if 'video' not in self._model.inputs:
                raise self._invalid(f'document {index}: {self._model.name} does not accept video')
            kind = 'video'
        else:
            allowed = ', '.join(
                sorted(IMAGE_MEDIA_TYPES | (VIDEO_MEDIA_TYPES if 'video' in self._model.inputs else set()))
            )
            raise self._invalid(f'document {index}: media type {media_type!r} is not supported; use one of {allowed}')

        if encodings and encodings[0] != encoding:
            raise self._invalid(
                f'document {index}: media is given as {"a URL" if encoding == "url" else "base64 data"}, but an '
                f'earlier input used {"a URL" if encodings[0] == "url" else "base64 data"}; one request must use '
                'either URLs or base64 data URLs for all its media, never both'
            )
        encodings.append(encoding)
        key = f'{kind}_{encoding}'
        return {'type': key, key: url}

    # Shared ---------------------------------------------------------------

    def _parse(self, config_class: type[_ConfigT], request: EmbedRequest[Any]) -> _ConfigT:
        options: object = request.options
        if options is None:
            return config_class()
        if isinstance(options, config_class):
            return options
        try:
            return config_class.model_validate(options)
        except ValidationError as error:
            problems = '; '.join(f'{".".join(str(p) for p in e["loc"])}: {e["msg"]}' for e in error.errors())
            raise self._invalid(f'invalid options ({problems})') from error

    def _cap(self, truncation: bool) -> int | None:
        """A text never costs more than the context length while the server truncates."""
        return self._model.context_tokens if truncation else None

    async def _plan(self, items: list[BatchItem], *, cap: int | None) -> list[range]:
        try:
            return await plan_batches(items, self._limits, cap=cap, count_tokens=self._count_tokens)
        except OversizeInputError as error:
            raise self._invalid(
                f'document {error.index} needs {error.tokens} tokens, but one request carries at most '
                f'{error.limit}; it cannot be sent as it is, so split it or shorten it'
            ) from error

    async def _count_tokens(self, texts: list[str]) -> list[int]:
        """Counts tokens with the SDK tokenizer, off the event loop (it may download on first use)."""
        try:
            encodings = await asyncio.to_thread(self._client.tokenize, texts, self._model.name)
        except Exception as error:  # noqa: BLE001 - whatever the download or tokenizer raised is the cause.
            raise GenkitError(
                status='UNAVAILABLE',
                message=(
                    f'voyageai {self._model.name}: cannot split this call into requests because the tokenizer '
                    f'failed to load ({type(error).__name__}: {error}); check access to huggingface.co, '
                    'or send fewer and smaller documents'
                ),
                cause=error,
            ) from error
        if len(encodings) != len(texts):
            raise GenkitError(
                status='INTERNAL',
                message=(
                    f'voyageai {self._model.name}: the tokenizer returned {len(encodings)} counts '
                    f'for {len(texts)} texts'
                ),
            )
        return [len(encoding) for encoding in encodings]

    def _check_dimension(self, dimension: int | None) -> None:
        if dimension is None or dimension in self._model.dimensions:
            return
        if len(self._model.dimensions) == 1:
            raise self._invalid(
                f'output_dimension {dimension} is not supported; {self._model.name} only produces '
                f'{self._model.dimensions[0]}-dimensional vectors'
            )
        supported = ', '.join(str(d) for d in sorted(self._model.dimensions))
        raise self._invalid(f'output_dimension {dimension} is not supported; use one of {supported}')

    def _vector(self, raw: Sequence[float]) -> list[float]:
        if len(raw) == 0:
            raise GenkitError(status='INTERNAL', message=f'voyageai {self._model.name}: returned an empty embedding')
        return [float(v) for v in raw]

    def _check_returned(self, returned: int, expected: int) -> None:
        if returned != expected:
            raise GenkitError(
                status='INTERNAL',
                message=(
                    f'voyageai {self._model.name}: returned {returned} results for {expected} inputs; '
                    'they cannot be matched to the documents'
                ),
            )

    def _invalid(self, message: str) -> GenkitError:
        return GenkitError(
            status='INVALID_ARGUMENT',
            message=f'voyageai {self._model.name}: {message}',
            reason=ErrorReason.INVALID_INPUT,
        )

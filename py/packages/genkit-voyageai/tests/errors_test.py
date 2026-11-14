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

"""Voyage error bodies reach the caller, whatever the status."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from genkit_voyageai._catalog import CATALOG
from genkit_voyageai._embedders import VoyageEmbedder
from genkit_voyageai._errors import to_genkit_error, voyage_detail
from voyageai import error as voyage_error

from genkit import GenkitError
from genkit.embedder import EmbedRequest


def _sdk_error(
    cls: type[voyage_error.VoyageError], status: int | None, body: dict[str, Any] | None, message: str
) -> Any:
    raw = None if body is None else json.dumps(body)
    return cls(message, raw, status, body)


@pytest.mark.parametrize(
    ('error', 'status', 'expected'),
    [
        # 4xx: the SDK already put the detail in its message; the body agrees.
        (
            _sdk_error(voyage_error.InvalidRequestError, 400, {'detail': 'Input is too long.'}, 'Input is too long.'),
            'INVALID_ARGUMENT',
            'Input is too long.',
        ),
        (
            _sdk_error(
                voyage_error.AuthenticationError,
                401,
                {'detail': 'Provided API key is invalid.'},
                'Provided API key is invalid.',
            ),
            'UNAUTHENTICATED',
            'Provided API key is invalid.',
        ),
        (
            _sdk_error(voyage_error.RateLimitError, 429, {'detail': 'Rate limit reached.'}, 'Rate limit reached.'),
            'RESOURCE_EXHAUSTED',
            'Rate limit reached.',
        ),
        # 5xx: the SDK replaces the body with a fixed sentence; the detail is only in the body.
        (
            _sdk_error(
                voyage_error.ServerError,
                500,
                {'detail': 'Model overloaded.'},
                'The server failed to process the request.',
            ),
            'INTERNAL',
            'Model overloaded.',
        ),
        (
            _sdk_error(
                voyage_error.ServiceUnavailableError,
                503,
                {'detail': 'Warming up.'},
                'The server is overloaded or not ready yet.',
            ),
            'UNAVAILABLE',
            'Warming up.',
        ),
        # Validation failures carry a list instead of a sentence.
        (
            _sdk_error(
                voyage_error.MalformedRequestError,
                422,
                {'detail': [{'loc': ['body', 'input'], 'msg': 'field required'}]},
                'x',
            ),
            'UNKNOWN',
            'field required',
        ),
        # No usable body: fall back to the SDK's message.
        (
            _sdk_error(voyage_error.ServerError, 500, None, 'The server failed to process the request.'),
            'INTERNAL',
            'The server failed to process the request.',
        ),
    ],
)
def test_status_and_detail_survive(error: voyage_error.VoyageError, status: str, expected: str) -> None:
    mapped = to_genkit_error(error, model='voyage-4')
    assert mapped.status == status
    assert expected in mapped.message
    assert mapped.message.startswith('voyageai voyage-4: ')


def test_detail_is_read_from_the_raw_body_when_the_sdk_did_not_parse_it() -> None:
    error = voyage_error.APIError('Invalid response object from API', '{"detail": "Bad thing."}', 403, None)
    assert voyage_detail(error) == 'Bad thing.'
    assert voyage_detail(voyage_error.APIError('plain', 'not json', 403, None)) == 'plain'


def test_failures_without_a_response_are_classified_by_type() -> None:
    assert to_genkit_error(voyage_error.Timeout('Request timed out'), model='m').status == 'DEADLINE_EXCEEDED'
    assert to_genkit_error(voyage_error.APIConnectionError('down'), model='m').status == 'UNAVAILABLE'
    assert to_genkit_error(voyage_error.InvalidRequestError('bad input'), model='m').status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_an_sdk_failure_surfaces_as_genkit_error_with_the_server_detail() -> None:
    client = MagicMock()
    client.embed = AsyncMock(
        side_effect=_sdk_error(
            voyage_error.InvalidRequestError, 400, {'detail': 'Too many tokens.'}, 'Too many tokens.'
        )
    )
    request = EmbedRequest(input=[{'content': [{'text': 'x'}]}])  # pyright: ignore[reportArgumentType]
    with pytest.raises(GenkitError, match='Too many tokens.') as raised:
        await VoyageEmbedder(CATALOG['voyage-4'], client).embed(request)
    assert raised.value.status == 'INVALID_ARGUMENT'

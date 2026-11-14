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

"""Turns Voyage SDK failures into GenkitError.

Voyage answers a rejected call with ``{"detail": "<text>"}``. The SDK reads that
field itself for 4xx statuses, but not for 5xx (it replaces the body with a
fixed sentence) and not when a 4xx body carries no usable ``detail``. This
module looks at the body first, so the server's own words reach the caller
whatever the status.
"""

import json
from typing import Any, cast

from voyageai import error as voyage_error

from genkit import GenkitError
from genkit.plugin_api import provider_error


def _detail_from(body: object) -> str | None:
    """Reads ``detail`` off a parsed error body, or None when there is none."""
    if not isinstance(body, dict):
        return None
    detail = cast(dict[str, Any], body).get('detail')
    if detail is None:
        return None
    if isinstance(detail, str):
        return detail or None
    # Validation failures carry a list of objects rather than a sentence.
    return json.dumps(detail, default=str)


def voyage_detail(error: voyage_error.VoyageError) -> str:
    """The server's explanation for a failed call.

    Prefers the ``detail`` of the response body, whether the SDK parsed it
    (``json_body``) or only kept the raw text (``http_body``), and falls back to
    the SDK's own message when the body has none.
    """
    detail = _detail_from(error.json_body)
    if detail is not None:
        return detail
    if isinstance(error.http_body, str) and error.http_body:
        try:
            detail = _detail_from(json.loads(error.http_body))
        except ValueError:
            detail = None
        if detail is not None:
            return detail
    return error.user_message or type(error).__name__


def to_genkit_error(error: voyage_error.VoyageError, *, model: str) -> GenkitError:
    """Maps a Voyage SDK error to a GenkitError that keeps Voyage's ``detail``.

    A response status maps the usual way (400 INVALID_ARGUMENT, 401
    UNAUTHENTICATED, 429 RESOURCE_EXHAUSTED, 5xx INTERNAL/UNAVAILABLE). Errors
    with no response are classified by type: a timeout is DEADLINE_EXCEEDED, a
    failed connection UNAVAILABLE, and a request the SDK itself refused
    INVALID_ARGUMENT.

    Args:
        error: What the Voyage SDK raised.
        model: The Voyage model the call was for.

    Returns:
        The GenkitError to raise.
    """
    message = f'voyageai {model}: {voyage_detail(error)}'
    if error.http_status is not None:
        return provider_error(error, http_status=error.http_status, headers=error.headers, message=message)
    if isinstance(error, voyage_error.Timeout):
        return provider_error(error, status='DEADLINE_EXCEEDED', message=message)
    if isinstance(error, voyage_error.APIConnectionError):
        return provider_error(error, status='UNAVAILABLE', message=message)
    if isinstance(error, (voyage_error.InvalidRequestError, voyage_error.MalformedRequestError)):
        return provider_error(error, status='INVALID_ARGUMENT', message=message)
    return provider_error(error, message=message)

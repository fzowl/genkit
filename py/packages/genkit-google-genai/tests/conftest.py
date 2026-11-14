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

"""Pytest configuration for the Google GenAI plugin tests."""

from collections.abc import Iterator

import pytest
from google.genai import _api_client


@pytest.fixture(autouse=True)
def _use_httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the SDK on its httpx transport.

    The SDK sends over aiohttp whenever it is installed, which bypasses the httpx
    fakes these tests install. Other workspace packages bring aiohttp in, so the
    tests must not depend on it being absent.
    """
    monkeypatch.setattr(_api_client, 'has_aiohttp', False)
    yield

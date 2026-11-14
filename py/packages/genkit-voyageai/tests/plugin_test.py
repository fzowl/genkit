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

"""Plugin setup: the key, the catalog, and the absence of a default model."""

import pytest
from genkit_voyageai import VoyageAI

from genkit import Genkit, GenkitError
from genkit.plugin_api import ActionKind


@pytest.mark.asyncio
async def test_missing_api_key_fails_at_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('VOYAGE_API_KEY', raising=False)
    with pytest.raises(GenkitError, match='VOYAGE_API_KEY') as raised:
        await VoyageAI().init()
    assert raised.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_blank_api_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('VOYAGE_API_KEY', '   ')
    with pytest.raises(GenkitError, match='no API key'):
        await VoyageAI().init()


@pytest.mark.asyncio
async def test_key_comes_from_the_option_or_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('VOYAGE_API_KEY', raising=False)
    await VoyageAI(api_key='pa-test').init()
    monkeypatch.setenv('VOYAGE_API_KEY', 'pa-test')
    await VoyageAI().init()


@pytest.mark.asyncio
async def test_only_catalog_models_resolve() -> None:
    plugin = VoyageAI(api_key='pa-test')
    assert await plugin.resolve(ActionKind.EMBEDDER, 'voyage-4') is not None
    assert await plugin.resolve(ActionKind.EMBEDDER, 'voyage-5') is None
    assert await plugin.resolve(ActionKind.EMBEDDER, '') is None
    assert await plugin.resolve(ActionKind.MODEL, 'voyage-4') is None
    # No reranker abstraction in Genkit Python, so rerank models are not embedders either.
    assert await plugin.resolve(ActionKind.EMBEDDER, 'rerank-2.5') is None


def test_embedder_ref_for_an_unknown_model_names_the_known_ones() -> None:
    with pytest.raises(GenkitError, match="unknown model 'voyage-5'.*voyage-4") as raised:
        VoyageAI.embedder('voyage-5')
    assert raised.value.status == 'INVALID_ARGUMENT'
    assert VoyageAI.embedder('voyageai/voyage-4').name == 'voyageai/voyage-4'


@pytest.mark.asyncio
async def test_unknown_model_is_not_found_through_genkit() -> None:
    ai = Genkit(plugins=[VoyageAI(api_key='pa-test')])
    with pytest.raises(GenkitError) as raised:
        await ai.embed(embedder='voyageai/voyage-5', content='x')
    assert raised.value.status == 'NOT_FOUND'


@pytest.mark.asyncio
async def test_config_for_the_wrong_model_family_is_rejected_through_genkit() -> None:
    ai = Genkit(plugins=[VoyageAI(api_key='pa-test')])
    # voyage-context-* has no truncation switch.
    with pytest.raises(GenkitError):
        await ai.embed(
            embedder='voyageai/voyage-context-4', content='x', config={'input_type': 'document', 'truncation': False}
        )

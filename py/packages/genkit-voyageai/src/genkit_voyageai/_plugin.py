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

"""Voyage AI plugin for Genkit."""

import os
from typing import Any

from voyageai.client_async import AsyncClient

from genkit import GenkitError
from genkit.embedder import Embedder, EmbedderInfo, EmbedderRef, EmbedderSupports, EmbedRequest, EmbedResponse
from genkit.plugin_api import Action, ActionKind, Plugin
from genkit_voyageai._catalog import CATALOG, VoyageModel
from genkit_voyageai._config import VoyageContextualizedEmbedConfig, VoyageEmbedConfig
from genkit_voyageai._embedders import VoyageEmbedder

VOYAGEAI_PLUGIN_NAME = 'voyageai'

DEFAULT_BASE_URL = 'https://api.voyageai.com'
"""Voyage's API host. The SDK appends ``/v1`` itself, see ``_sdk_base_url``."""

_PREFIX = f'{VOYAGEAI_PLUGIN_NAME}/'


def _action_name(model: str) -> str:
    return f'{_PREFIX}{model}'


def _sdk_base_url(base_url: str) -> str:
    """The SDK takes the versioned root; this plugin takes the host, like the other SDKs do."""
    root = base_url.rstrip('/')
    return root if root.endswith('/v1') else f'{root}/v1'


def _config_class(model: VoyageModel) -> type[VoyageEmbedConfig] | type[VoyageContextualizedEmbedConfig]:
    return VoyageContextualizedEmbedConfig if model.kind == 'contextualized' else VoyageEmbedConfig


def _known_models() -> str:
    return ', '.join(sorted(CATALOG))


def _embedder_info(model: VoyageModel) -> EmbedderInfo:
    return EmbedderInfo(
        label=f'Voyage AI - {model.name}',
        dimensions=model.default_dimension,
        supports=EmbedderSupports(input=list(model.inputs)),
    )


class VoyageAI(Plugin):
    """Voyage AI embedders for Genkit.

    Serves the text embedders (``voyage-4`` and friends), the contextualized
    embedders (``voyage-context-*``) and the multimodal embedders
    (``voyage-multimodal-*``) as ``voyageai/<model>``. There is no default
    model: name one on every call.

    Example:
        ```python
        from genkit import Genkit
        from genkit_voyageai import VoyageAI

        ai = Genkit(plugins=[VoyageAI()])

        vectors = await ai.embed(
            embedder=VoyageAI.embedder('voyage-4'),
            content='Smoked salmon tartine',
            config={'input_type': 'document', 'output_dimension': 256},
        )
        print(len(vectors[0].embedding))
        # => 256
        ```
    """

    name = VOYAGEAI_PLUGIN_NAME

    @classmethod
    def embedder(
        cls,
        name: str,
        *,
        config: VoyageEmbedConfig | VoyageContextualizedEmbedConfig | None = None,
    ) -> EmbedderRef[Any]:
        """EmbedderRef for ``ai.embed()``, e.g. ``VoyageAI.embedder('voyage-4')``.

        Args:
            name: A catalog model name, with or without the ``voyageai/`` prefix.
            config: Default settings for calls made through this ref;
                ``VoyageContextualizedEmbedConfig`` for ``voyage-context-*``,
                ``VoyageEmbedConfig`` for every other model.

        Returns:
            A ref named ``voyageai/<name>``.

        Raises:
            GenkitError: INVALID_ARGUMENT when ``name`` is not a catalog model.
        """
        if not isinstance(name, str):
            raise GenkitError(status='INVALID_ARGUMENT', message='VoyageAI.embedder: model name must be a string.')
        local = name.removeprefix(_PREFIX)
        model = CATALOG.get(local)
        if model is None:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'VoyageAI.embedder: unknown model {local!r}; known models: {_known_models()}.',
            )
        return EmbedderRef(name=_action_name(model.name), config_schema=_config_class(model), config=config)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initializes the Voyage AI plugin.

        Args:
            api_key: Voyage API key. Defaults to the ``VOYAGE_API_KEY``
                environment variable. A missing key fails at ``init``.
            base_url: API host, defaulting to ``https://api.voyageai.com``.
            timeout: Per-request timeout in seconds. Defaults to the SDK's.
        """
        self._api_key = api_key
        self._base_url = base_url or DEFAULT_BASE_URL
        self._timeout = timeout
        self._client: AsyncClient | None = None

    def _get_client(self) -> AsyncClient:
        """The shared SDK client, built on first use.

        The key is passed explicitly so the SDK's own lookups (a module-level
        key, ``VOYAGE_API_KEY_PATH``) and its key-prefix choice of host never
        apply.
        """
        if self._client is None:
            api_key = (self._api_key or os.environ.get('VOYAGE_API_KEY') or '').strip()
            if not api_key:
                raise GenkitError(
                    status='FAILED_PRECONDITION',
                    message=(
                        'VoyageAI: no API key. Pass VoyageAI(api_key=...) or set the VOYAGE_API_KEY environment '
                        'variable; keys come from https://dash.voyageai.com.'
                    ),
                )
            self._client = AsyncClient(
                api_key=api_key,
                base_url=_sdk_base_url(self._base_url),
                timeout=self._timeout,
            )
        return self._client

    async def init(self) -> list[Action]:
        """Initializes the plugin.

        Builds the client so a missing API key fails at startup rather than on
        the first embed call.

        Returns:
            An empty list; embedders are created by ``resolve`` and ``list_actions``.

        Raises:
            GenkitError: FAILED_PRECONDITION when no API key is configured.
        """
        self._get_client()
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolves a catalog embedder; anything else is declined.

        Args:
            action_type: The kind of action asked for.
            name: The model name without the ``voyageai/`` prefix.

        Returns:
            The embedder, or None for any other kind or a name outside the
            catalog. Nothing falls back to another model.
        """
        if action_type != ActionKind.EMBEDDER:
            return None
        model = CATALOG.get(name)
        return None if model is None else self._create_embedder_action(model)

    def _create_embedder_action(self, model: VoyageModel) -> Action:
        async def _run(request: EmbedRequest[Any]) -> EmbedResponse:
            return await VoyageEmbedder(model, self._get_client()).embed(request)

        return Embedder(
            _action_name(model.name),
            _run,
            config_schema=_config_class(model),
            info=_embedder_info(model),
        )

    async def list_actions(self) -> list[Action]:
        """Lists every catalog embedder for the Dev UI.

        Returns:
            One embedder action per catalog model.
        """
        return [self._create_embedder_action(model) for model in CATALOG.values()]

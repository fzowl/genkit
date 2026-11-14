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

"""Voyage AI plugin for Genkit.

Embedders only: text (``voyage-4``, ``voyage-code-4``, ...), contextualized
(``voyage-context-4``, ``voyage-context-3``) and multimodal
(``voyage-multimodal-3.5``, ``voyage-multimodal-3``). There is no default
model, so every call names one as ``voyageai/<model>``.

Prerequisites:
    - A Voyage API key from https://dash.voyageai.com, passed as
      ``VoyageAI(api_key=...)`` or set as ``VOYAGE_API_KEY``.

Example:
    ```python
    from genkit import Genkit
    from genkit_voyageai import VoyageAI

    ai = Genkit(plugins=[VoyageAI()])

    # 1. Embed a document at a reduced width
    vectors = await ai.embed(
        embedder=VoyageAI.embedder('voyage-4'),
        content='Smoked salmon tartine',
        config={'input_type': 'document', 'output_dimension': 256},
    )
    print(len(vectors[0].embedding))
    # => 256
    ```

See Also:
    - Voyage AI embeddings: https://docs.voyageai.com/docs/embeddings
"""

from genkit_voyageai._config import VoyageContextualizedEmbedConfig, VoyageEmbedConfig
from genkit_voyageai._plugin import VoyageAI

__all__ = [
    'VoyageAI',
    'VoyageContextualizedEmbedConfig',
    'VoyageEmbedConfig',
]

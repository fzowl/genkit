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

"""Per-request settings for Voyage AI embedders."""

from typing import Literal

from pydantic import Field

from genkit.plugin_api import GenkitConfig

InputType = Literal['document', 'query']


class VoyageEmbedConfig(GenkitConfig):
    """Settings for the text and multimodal embedders.

    Pass an instance to ``VoyageAI.embedder(name, config=...)``, or a dict in
    snake_case or camelCase to ``ai.embed(config=...)``. Unknown keys raise
    ``GenkitError(INVALID_ARGUMENT)``.
    """

    input_type: InputType | None = Field(
        default=None,
        description=(
            "Whether the input is a 'document' to be searched or a 'query' to search with. "
            'Voyage prepends a different prompt to each. Omitted from the request when unset.'
        ),
    )
    output_dimension: int | None = Field(
        default=None,
        description="Width of the returned vectors. Must be one of the model's supported dimensions.",
    )
    truncation: bool | None = Field(
        default=None,
        description='Truncate inputs longer than the model context instead of failing. Voyage defaults to true.',
    )


class VoyageContextualizedEmbedConfig(GenkitConfig):
    """Settings for the contextualized embedders (``voyage-context-*``).

    ``input_type`` is required: 'document' sends each input through automatic
    chunking, 'query' embeds each input as one query. The API has no
    truncation switch for these models.
    """

    input_type: InputType | None = Field(
        default=None,
        description="Required. 'document' auto-chunks each input; 'query' embeds each input as a single query.",
    )
    output_dimension: int | None = Field(
        default=None,
        description="Width of the returned vectors. Must be one of the model's supported dimensions.",
    )

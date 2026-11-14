/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import type { EmbedderReference, RerankerReference } from 'genkit';
import { genkitPluginV2, type GenkitPluginV2 } from 'genkit/plugin';
import {
  CONTEXTUALIZED_EMBEDDING_MODELS,
  MULTIMODAL_EMBEDDING_MODELS,
  RERANKER_MODELS,
  TEXT_EMBEDDING_MODELS,
  type ContextualizedEmbeddingModelName,
  type MultimodalEmbeddingModelName,
  type RerankerModelName,
  type TextEmbeddingModelName,
} from './catalog.js';
import {
  ContextualizedEmbedderConfigSchema,
  MultimodalEmbedderConfigSchema,
  TextEmbedderConfigSchema,
  contextualizedEmbedderRef,
  defineContextualizedEmbedder,
  defineMultimodalEmbedder,
  defineTextEmbedder,
  embedderReference,
  multimodalEmbedderRef,
  textEmbedderRef,
} from './embedders.js';
import {
  RerankerConfigSchema,
  defineReranker,
  rerankerReference,
} from './reranker.js';
import { createClient } from './shared.js';
import { createTokenCounter } from './tokenizer.js';
import type { VoyageAIPluginOptions } from './types.js';

export {
  CONTEXTUALIZED_EMBEDDING_MODELS,
  MULTIMODAL_EMBEDDING_MODELS,
  RERANKER_MODELS,
  TEXT_EMBEDDING_MODELS,
  type ContextualizedEmbeddingModelName,
  type EmbedderModelName,
  type MultimodalEmbeddingModelName,
  type RerankerModelName,
  type TextEmbeddingModelName,
} from './catalog.js';
export {
  CONTEXTUALIZED_CHUNK_SIZE,
  ContextualizedEmbedderConfigSchema,
  MultimodalEmbedderConfigSchema,
  TextEmbedderConfigSchema,
  type ContextualizedChunkMetadata,
  type ContextualizedEmbedderConfig,
  type MultimodalEmbedderConfig,
  type TextEmbedderConfig,
} from './embedders.js';
export { RerankerConfigSchema, type RerankerConfig } from './reranker.js';
export { extractVoyageErrorDetail } from './shared.js';
export type { VoyageAIPluginOptions } from './types.js';

function voyageaiPlugin(options: VoyageAIPluginOptions = {}): GenkitPluginV2 {
  return genkitPluginV2({
    name: 'voyageai',
    init: () => {
      const client = createClient(options);
      const countTokens =
        options.countTokens ?? createTokenCounter(options.fetch);
      return [
        ...(Object.keys(TEXT_EMBEDDING_MODELS) as TextEmbeddingModelName[]).map(
          (name) => defineTextEmbedder(name, client, countTokens)
        ),
        ...(
          Object.keys(
            CONTEXTUALIZED_EMBEDDING_MODELS
          ) as ContextualizedEmbeddingModelName[]
        ).map((name) =>
          defineContextualizedEmbedder(name, client, countTokens)
        ),
        ...(
          Object.keys(
            MULTIMODAL_EMBEDDING_MODELS
          ) as MultimodalEmbeddingModelName[]
        ).map((name) => defineMultimodalEmbedder(name, client, countTokens)),
        ...(Object.keys(RERANKER_MODELS) as RerankerModelName[]).map((name) =>
          defineReranker(name, client)
        ),
      ];
    },
  });
}

/** The `voyageai` plugin together with its model reference helpers. */
export type VoyageAIPlugin = {
  (options?: VoyageAIPluginOptions): GenkitPluginV2;
  /**
   * Reference to a Voyage text, contextualized or multimodal embedding model.
   * There is no default model: the model must always be named.
   */
  embedder(
    name: TextEmbeddingModelName
  ): EmbedderReference<typeof TextEmbedderConfigSchema>;
  embedder(
    name: ContextualizedEmbeddingModelName
  ): EmbedderReference<typeof ContextualizedEmbedderConfigSchema>;
  embedder(
    name: MultimodalEmbeddingModelName
  ): EmbedderReference<typeof MultimodalEmbedderConfigSchema>;
  embedder(name: string): EmbedderReference;
  /** Reference to a Voyage reranker model. There is no default model. */
  reranker(
    name: RerankerModelName
  ): RerankerReference<typeof RerankerConfigSchema>;
  reranker(name: string): RerankerReference<typeof RerankerConfigSchema>;
};

/**
 * Voyage AI plugin for Genkit: text, contextualized and multimodal embedders
 * and rerankers.
 *
 * ```ts
 * import { genkit } from 'genkit';
 * import { voyageai } from '@genkit-ai/voyageai';
 *
 * const ai = genkit({ plugins: [voyageai()] }); // reads VOYAGE_API_KEY
 *
 * const embeddings = await ai.embed({
 *   embedder: voyageai.embedder('voyage-4'),
 *   content: 'hello',
 *   options: { inputType: 'document' },
 * });
 * ```
 */
export const voyageai = voyageaiPlugin as VoyageAIPlugin;
voyageai.embedder = ((name: string) => embedderReference(name)) as never;
voyageai.reranker = ((name: string) => rerankerReference(name)) as never;

export { contextualizedEmbedderRef, multimodalEmbedderRef, textEmbedderRef };
export default voyageai;

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
/** Output dimensions of the Matryoshka-capable Voyage models; the first one is the default. */
const FLEXIBLE_DIMENSIONS = [1024, 2048, 512, 256] as const;
const FIXED_DIMENSIONS = [1024] as const;

/** Documented limits of a Voyage embedding model. */
export interface EmbeddingModelSpec {
  /** Supported output dimensions; the first entry is the model default. */
  dimensions: readonly number[];
  /** Context length of a single input, in tokens. */
  contextTokens: number;
  /**
   * Total tokens Voyage accepts per request. Enforced by the API, kept here as
   * documentation. Left undefined when Voyage does not document a ceiling.
   */
  maxTokensPerRequest?: number;
}

/** Maximum number of inputs Voyage accepts in one embeddings request. */
export const MAX_INPUTS_PER_REQUEST = 1000;

/** Text embedding models served by `POST /v1/embeddings`. */
export const TEXT_EMBEDDING_MODELS = {
  'voyage-4-large': {
    dimensions: FLEXIBLE_DIMENSIONS,
    contextTokens: 32_000,
    maxTokensPerRequest: 120_000,
  },
  'voyage-4': {
    dimensions: FLEXIBLE_DIMENSIONS,
    contextTokens: 32_000,
    maxTokensPerRequest: 320_000,
  },
  'voyage-4-lite': {
    dimensions: FLEXIBLE_DIMENSIONS,
    contextTokens: 32_000,
    maxTokensPerRequest: 1_000_000,
  },
  // Voyage documents no per-request token ceiling for voyage-code-4.
  'voyage-code-4': {
    dimensions: FLEXIBLE_DIMENSIONS,
    contextTokens: 32_000,
  },
  'voyage-3-large': {
    dimensions: FLEXIBLE_DIMENSIONS,
    contextTokens: 32_000,
    maxTokensPerRequest: 120_000,
  },
  'voyage-3.5': {
    dimensions: FLEXIBLE_DIMENSIONS,
    contextTokens: 32_000,
    maxTokensPerRequest: 320_000,
  },
  'voyage-3.5-lite': {
    dimensions: FLEXIBLE_DIMENSIONS,
    contextTokens: 32_000,
    maxTokensPerRequest: 1_000_000,
  },
  'voyage-code-3': {
    dimensions: FLEXIBLE_DIMENSIONS,
    contextTokens: 32_000,
    maxTokensPerRequest: 120_000,
  },
  'voyage-finance-2': {
    dimensions: FIXED_DIMENSIONS,
    contextTokens: 32_000,
    maxTokensPerRequest: 120_000,
  },
  'voyage-law-2': {
    dimensions: FIXED_DIMENSIONS,
    contextTokens: 16_000,
    maxTokensPerRequest: 120_000,
  },
} as const satisfies Record<string, EmbeddingModelSpec>;
export type TextEmbeddingModelName = keyof typeof TEXT_EMBEDDING_MODELS;

/** Contextualized chunk embedding models served by `POST /v1/contextualizedembeddings`. */
export const CONTEXTUALIZED_EMBEDDING_MODELS = {
  'voyage-context-4': {
    dimensions: FLEXIBLE_DIMENSIONS,
    contextTokens: 32_000,
    maxTokensPerRequest: 120_000,
  },
  'voyage-context-3': {
    dimensions: FLEXIBLE_DIMENSIONS,
    contextTokens: 32_000,
    maxTokensPerRequest: 120_000,
  },
} as const satisfies Record<string, EmbeddingModelSpec>;
export type ContextualizedEmbeddingModelName =
  keyof typeof CONTEXTUALIZED_EMBEDDING_MODELS;

/** Documented limits of a Voyage multimodal embedding model. */
export interface MultimodalModelSpec extends EmbeddingModelSpec {
  /** Whether the model accepts video inputs. */
  video: boolean;
}

/** Multimodal embedding models served by `POST /v1/multimodalembeddings`. */
export const MULTIMODAL_EMBEDDING_MODELS = {
  'voyage-multimodal-3.5': {
    dimensions: FLEXIBLE_DIMENSIONS,
    contextTokens: 32_000,
    maxTokensPerRequest: 320_000,
    video: true,
  },
  'voyage-multimodal-3': {
    dimensions: FIXED_DIMENSIONS,
    contextTokens: 32_000,
    maxTokensPerRequest: 320_000,
    video: false,
  },
} as const satisfies Record<string, MultimodalModelSpec>;
export type MultimodalEmbeddingModelName =
  keyof typeof MULTIMODAL_EMBEDDING_MODELS;

/** Reranker models served by `POST /v1/rerank`. */
export const RERANKER_MODELS = {
  'rerank-3': { contextTokens: 32_000 },
  'rerank-3-lite': { contextTokens: 32_000 },
  'rerank-2.5': { contextTokens: 32_000 },
  'rerank-2.5-lite': { contextTokens: 32_000 },
} as const;
export type RerankerModelName = keyof typeof RERANKER_MODELS;

/** Maximum number of documents Voyage accepts in one rerank request. */
export const MAX_RERANK_DOCUMENTS = 1000;

/** Every embedder model name this plugin defines. */
export type EmbedderModelName =
  | TextEmbeddingModelName
  | ContextualizedEmbeddingModelName
  | MultimodalEmbeddingModelName;

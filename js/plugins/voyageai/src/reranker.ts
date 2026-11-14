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
import { z, type RerankerAction } from 'genkit';
import { reranker as pluginReranker } from 'genkit/plugin';
import { RankedDocument, rerankerRef } from 'genkit/reranker';
import type { VoyageAIClient } from 'voyageai';
import {
  MAX_RERANK_DOCUMENTS,
  RERANKER_MODELS,
  type RerankerModelName,
} from './catalog.js';
import { callVoyage, invalidArgument, parseConfig } from './shared.js';

const NAMESPACE = 'voyageai';

/** Options of the Voyage rerankers. */
export const RerankerConfigSchema = z
  .object({
    topK: z
      .number()
      .int()
      .positive()
      .optional()
      .describe(
        'Number of most relevant documents to return. All documents are returned, ordered by relevance, when unset.'
      ),
    truncation: z
      .boolean()
      .optional()
      .describe(
        'Truncate the query and documents that exceed the context length. Voyage truncates by default; false makes over-length input an error.'
      ),
  })
  .strict();
export type RerankerConfig = z.infer<typeof RerankerConfigSchema>;

/** Returns the reranker reference for a known Voyage reranker model, or throws for an unknown one. */
export function rerankerReference(name: string) {
  if (!(name in RERANKER_MODELS)) {
    throw invalidArgument(
      `Unknown Voyage AI reranker model '${name}'. Known models: ${Object.keys(RERANKER_MODELS).join(', ')}.`
    );
  }
  return rerankerRef({
    name: `${NAMESPACE}/${name}`,
    configSchema: RerankerConfigSchema,
    info: { label: `Voyage AI - ${name}`, supports: { media: false } },
  });
}

/** Defines the reranker action of a Voyage reranker model. */
export function defineReranker(
  name: RerankerModelName,
  client: VoyageAIClient
): RerankerAction {
  const ref = rerankerReference(name);
  return pluginReranker(
    {
      name: ref.name,
      configSchema: ref.configSchema,
      info: ref.info,
    },
    async (query, documents, options) => {
      const config = parseConfig(RerankerConfigSchema, options, ref.name);
      if (query.media.length > 0) {
        throw invalidArgument(
          `${ref.name} reranks text only; the query contains media.`
        );
      }
      if (documents.length > MAX_RERANK_DOCUMENTS) {
        throw invalidArgument(
          `${ref.name} accepts at most ${MAX_RERANK_DOCUMENTS} documents, got ${documents.length}.`
        );
      }
      if (documents.length === 0) return { documents: [] };
      const texts = documents.map((doc, i) => {
        if (doc.media.length > 0) {
          throw invalidArgument(
            `${ref.name} reranks text only; document ${i} contains media.`
          );
        }
        return doc.text;
      });
      const response = await callVoyage('rerank request', () =>
        client.rerank({
          query: query.text,
          documents: texts,
          model: name,
          topK: config.topK,
          returnDocuments: true,
          truncation: config.truncation,
        })
      );
      const items = response.data ?? [];
      const expected = Math.min(config.topK ?? texts.length, texts.length);
      if (items.length !== expected) {
        throw new Error(
          `${ref.name}: Voyage returned ${items.length} results, expected ${expected}.`
        );
      }
      return {
        documents: items
          .map((item) => {
            if (
              item.index === undefined ||
              item.index < 0 ||
              item.index >= documents.length ||
              typeof item.relevanceScore !== 'number'
            ) {
              throw new Error(
                `${ref.name}: Voyage returned a result without a valid index or relevance score.`
              );
            }
            const original = documents[item.index];
            return new RankedDocument({
              content: original.content,
              metadata: {
                ...original.metadata,
                score: item.relevanceScore,
              },
            });
          })
          .sort((a, b) => b.score() - a.score()),
      };
    }
  ) as RerankerAction;
}

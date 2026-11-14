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
// Live tests against the real Voyage API. Skipped without VOYAGE_API_KEY.
import * as assert from 'assert';
import { genkit } from 'genkit';
import type { Embedding } from 'genkit/embedder';
import { describe, it } from 'node:test';
import { voyageai } from '../src/index.js';

const png =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

describe('voyageai live', { skip: !process.env.VOYAGE_API_KEY }, () => {
  const ai = genkit({ plugins: [voyageai()] });

  it('embeds text with a smaller dimension', async () => {
    const [e] = await ai.embedMany({
      embedder: voyageai.embedder('voyage-4-lite'),
      content: ['hello'],
      options: { inputType: 'query', outputDimension: 256 },
    });
    assert.equal(e.embedding.length, 256);
  });

  it('returns chunk texts for auto-chunked documents', async () => {
    const out = (await ai.embedMany({
      embedder: voyageai.embedder('voyage-context-4'),
      content: ['Cats purr. Dogs bark.', 'Rome is in Italy.'],
      options: { inputType: 'document', outputDimension: 256 },
    })) as Embedding[];
    assert.equal(out.length >= 2, true);
    assert.equal(typeof out[0].metadata?.text, 'string');
    assert.equal(out[0].embedding.length, 256);
  });

  it('embeds mixed text and a base64 image', async () => {
    const [e] = await ai.embedMany({
      embedder: voyageai.embedder('voyage-multimodal-3.5'),
      content: [{ content: [{ text: 'a pixel' }, { media: { url: png } }] }],
      options: { inputType: 'document' },
    });
    assert.equal(e.embedding.length, 1024);
  });

  it('reranks with top_k', async () => {
    const ranked = await ai.rerank({
      reranker: voyageai.reranker('rerank-3'),
      query: 'flying animals',
      documents: ['cats', 'birds', 'rocks'].map((text) => ({
        content: [{ text }],
      })),
      options: { topK: 2 },
    });
    assert.equal(ranked.length, 2);
    assert.equal(ranked[0].text, 'birds');
  });
});

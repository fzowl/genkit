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
import * as assert from 'assert';
import { describe, it } from 'node:test';
import { inputCosts, packBatches } from '../src/batching.js';
import { voyageai } from '../src/index.js';
import { embeddingsBody, setup } from './helpers.js';

const limits = { maxInputs: 1000, maxTokens: 100 };

describe('packBatches', () => {
  it('keeps order and fills batches up to the token limit', () => {
    assert.deepEqual(packBatches([40, 40, 40, 20, 100, 1], limits, 'x'), [
      { start: 0, end: 2 },
      { start: 2, end: 4 },
      { start: 4, end: 5 },
      { start: 5, end: 6 },
    ]);
    assert.deepEqual(packBatches([50, 50, 1], limits, 'x'), [
      { start: 0, end: 2 },
      { start: 2, end: 3 },
    ]);
  });

  it('splits by input count and, without a token ceiling, only by count', () => {
    const batches = packBatches(
      new Array(2500).fill(1_000_000),
      {
        maxInputs: 1000,
      },
      'x'
    );
    assert.deepEqual(
      batches.map((b) => b.end - b.start),
      [1000, 1000, 500]
    );
  });

  it('refuses an input that cannot fit any request', () => {
    assert.throws(
      () => packBatches([5, 101], limits, 'voyageai/voyage-context-4'),
      /input 1 has 101 tokens, more than the 100 tokens/
    );
  });

  it('returns no batch for no input', () => {
    assert.deepEqual(packBatches([], limits, 'x'), []);
  });
});

describe('inputCosts', () => {
  const base = { model: 'voyage-4', contextTokens: 1000, truncates: true };

  it('does not count tokens when the byte bound fits one request', async () => {
    let calls = 0;
    const costs = await inputCosts([{ texts: ['abc'] }, { texts: ['de'] }], {
      ...base,
      maxTokens: 100,
      countTokens: async () => (calls++, []),
    });
    assert.deepEqual(costs, [5, 4]);
    assert.equal(calls, 0);
  });

  it('bounds tokens on the NFC form, which can be longer than the raw text', async () => {
    const text = '\uFB2C'.repeat(100); // 300 raw bytes, 600 bytes once normalized
    assert.equal(Buffer.byteLength(text, 'utf8'), 300);
    assert.equal(Buffer.byteLength(text.normalize('NFC'), 'utf8'), 600);
    let calls = 0;
    const costs = await inputCosts([{ texts: [text] }], {
      ...base,
      contextTokens: 10_000,
      maxTokens: 400, // fits the raw bytes (302) but not the NFC bytes (602)
      countTokens: async () => (calls++, [500]),
    });
    assert.equal(calls, 1);
    assert.deepEqual(costs, [500]);
  });

  it('counts the text segments of an input separately and sums them', async () => {
    let segments: string[] = [];
    const costs = await inputCosts(
      [{ texts: ['a'.repeat(300), 'b'.repeat(300)] }, { texts: [] }],
      {
        ...base,
        contextTokens: 10_000,
        maxTokens: 100,
        countTokens: async (_m, texts) => ((segments = texts), [30, 40]),
      }
    );
    assert.deepEqual(segments, ['a'.repeat(300), 'b'.repeat(300)]);
    assert.deepEqual(costs, [70, 0]);
  });

  it('counts nothing without a documented ceiling', async () => {
    let calls = 0;
    const costs = await inputCosts([{ texts: ['x'.repeat(10_000)] }], {
      ...base,
      countTokens: async () => (calls++, []),
    });
    assert.equal(costs, undefined);
    assert.equal(calls, 0);
  });

  it('counts exactly when the bound does not fit, capped at the context length', async () => {
    const costs = await inputCosts(
      [{ texts: ['x'.repeat(500)] }, { texts: ['y'.repeat(500)] }],
      {
        ...base,
        maxTokens: 600,
        countTokens: async () => [10, 5000],
      }
    );
    assert.deepEqual(costs, [10, 1000]);
    const uncapped = await inputCosts([{ texts: ['x'.repeat(2000)] }], {
      ...base,
      maxTokens: 600,
      truncates: false,
      countTokens: async () => [5000],
    });
    assert.deepEqual(uncapped, [5000]);
  });

  it('charges inputs with media the context length and counts only the others', async () => {
    let counted: string[] = [];
    const costs = await inputCosts(
      [{ texts: ['hello'], hasMedia: true }, { texts: ['x'.repeat(2000)] }],
      {
        ...base,
        maxTokens: 1500,
        countTokens: async (_m, texts) => ((counted = texts), [7]),
      }
    );
    assert.deepEqual(costs, [1000, 7]);
    assert.deepEqual(counted, ['x'.repeat(2000)]);
  });

  it('fails closed when the tokenizer cannot be loaded', async () => {
    await assert.rejects(
      inputCosts([{ texts: ['x'.repeat(2000)] }], {
        ...base,
        maxTokens: 100,
        countTokens: async () => {
          throw new Error('offline');
        },
      }),
      /Could not count tokens with the voyage-4 tokenizer.*offline/
    );
  });
});

describe('batched requests', () => {
  it('splits more than 1000 inputs into several requests, in order', async () => {
    const { ai, requests } = setup((r) => ({
      body: embeddingsBody(r.body.input.length),
    }));
    const out = await ai.embedMany({
      embedder: voyageai.embedder('voyage-4-lite'),
      content: Array.from({ length: 2300 }, () => 'a'),
    });
    assert.deepEqual(
      requests.map((r) => r.body.input.length),
      [1000, 1000, 300]
    );
    assert.equal(out.length, 2300);
  });

  it('splits by tokens with the tokenizer when the byte bound does not fit', async () => {
    const seen: string[][] = [];
    const { ai, requests } = setup(
      (r) => ({ body: embeddingsBody(r.body.input.length) }),
      {
        countTokens: async (_model, texts) => {
          seen.push(texts);
          // 100k tokens per text: voyage-4-large takes 120k per request.
          return texts.map(() => 100_000);
        },
      }
    );
    const text = 'x'.repeat(400_000);
    const out = await ai.embedMany({
      embedder: voyageai.embedder('voyage-4-large'),
      content: [text, text, text],
      options: { truncation: false },
    });
    assert.equal(seen.length, 1);
    assert.deepEqual(
      requests.map((r) => r.body.input.length),
      [1, 1, 1]
    );
    assert.equal(out.length, 3);
  });

  it('never tokenizes a small request', async () => {
    const { ai, requests } = setup(
      (r) => ({ body: embeddingsBody(r.body.input.length) }),
      {
        countTokens: async () => {
          throw new Error('tokenizer must not be used');
        },
      }
    );
    await ai.embedMany({
      embedder: voyageai.embedder('voyage-4'),
      content: ['small', 'texts'],
    });
    assert.equal(requests.length, 1);
  });

  it('keeps document indexes global across contextualized batches', async () => {
    const { ai, requests } = setup(
      (r) => ({
        body: {
          data: r.body.inputs.map((_: unknown, i: number) => ({
            index: i,
            data: [{ embedding: [1, 2], index: 0, text: `t${i}` }],
          })),
          usage: { total_tokens: 1 },
        },
      }),
      { countTokens: async (_m, texts) => texts.map(() => 70_000) }
    );
    const text = 'x'.repeat(300_000);
    const out = await ai.embedMany({
      embedder: voyageai.embedder('voyage-context-4'),
      content: [text, text, text],
      options: { inputType: 'document' },
    });
    assert.deepEqual(
      requests.map((r) => r.body.inputs.length),
      [1, 1, 1]
    );
    assert.deepEqual(
      out.map((e) => (e as any).metadata.documentIndex),
      [0, 1, 2]
    );
    assert.equal(requests[1].body.chunk_size, 32000);
    assert.equal(requests[1].body.enable_auto_chunking, true);
  });

  it('rejects a contextualized document that cannot fit any request', async () => {
    const { ai, requests } = setup(() => ({ body: {} }), {
      countTokens: async (_m, texts) => texts.map(() => 130_000),
    });
    await assert.rejects(
      ai.embedMany({
        embedder: voyageai.embedder('voyage-context-3'),
        content: ['x'.repeat(600_000)],
        options: { inputType: 'document' },
      }),
      /input 0 has 130000 tokens, more than the 120000 tokens/
    );
    assert.equal(requests.length, 0);
  });

  it('batches by count only for voyage-code-4, which has no documented ceiling', async () => {
    const { ai, requests } = setup(
      (r) => ({ body: embeddingsBody(r.body.input.length) }),
      {
        countTokens: async () => {
          throw new Error('nothing to count');
        },
      }
    );
    await ai.embedMany({
      embedder: voyageai.embedder('voyage-code-4'),
      content: ['x'.repeat(2_000_000), 'y'],
    });
    assert.equal(requests.length, 1);
  });

  it('treats inputs with media as a full context for multimodal batches', async () => {
    const png =
      'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
    const { ai, requests } = setup((r) => ({
      body: embeddingsBody(r.body.inputs.length),
    }));
    // 320,000 tokens per request and 32,000 per image input: 10 per request.
    await ai.embedMany({
      embedder: voyageai.embedder('voyage-multimodal-3.5'),
      content: Array.from({ length: 25 }, () => ({
        content: [{ media: { url: png } }],
      })),
    });
    assert.deepEqual(
      requests.map((r) => r.body.inputs.length),
      [10, 10, 5]
    );
  });
});

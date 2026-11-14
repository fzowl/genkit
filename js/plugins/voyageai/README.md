# Voyage AI plugin for Genkit

`@genkit-ai/voyageai` adds [Voyage AI](https://docs.voyageai.com/) text,
contextualized and multimodal embedders and rerankers to
[Genkit](https://github.com/genkit-ai/genkit). It uses the official
[`voyageai`](https://www.npmjs.com/package/voyageai) SDK.

## Installation

```bash
npm install @genkit-ai/voyageai
```

## Setup

```ts
import { genkit } from 'genkit';
import { voyageai } from '@genkit-ai/voyageai';

// Reads the VOYAGE_API_KEY environment variable. Pass `apiKey` to override.
const ai = genkit({ plugins: [voyageai()] });
```

There is **no default model**: every embedder and reranker is named explicitly,
and an unknown name is an error.

## Text embeddings

`voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-4`,
`voyage-3-large`, `voyage-3.5`, `voyage-3.5-lite`, `voyage-code-3`
(dimensions 1024 by default, or 2048 / 512 / 256), `voyage-finance-2`
(32k context) and `voyage-law-2` (16k context), both fixed at 1024 dimensions.

```ts
const embeddings = await ai.embedMany({
  embedder: voyageai.embedder('voyage-4'),
  content: ['first text', 'second text'],
  options: { inputType: 'document', outputDimension: 512 },
});
```

Options: `inputType` (`'document'` or `'query'`, omitted from the request when
unset), `outputDimension` (validated against the model, sent only when set) and
`truncation`. A request takes at most 1000 inputs.

## Contextualized chunk embeddings

`voyage-context-4` and `voyage-context-3` embed each chunk with the context of
its whole document. `inputType` is required.

- `'document'`: each input is one whole document. Voyage chunks it
  (`enable_auto_chunking` with 32,000-token chunks) and the result holds one
  embedding per chunk, in document and chunk order. Every embedding's
  `metadata` is `{ documentIndex, chunkIndex, text }`, where `text` is the chunk.
  A document must be a single text part: pre-chunked documents are rejected.
- `'query'`: each input is a query, embedded without chunking, one embedding per
  query. There is no chunk `text`.

```ts
const chunks = await ai.embedMany({
  embedder: voyageai.embedder('voyage-context-4'),
  content: [longDocument],
  options: { inputType: 'document' },
});
// chunks[i].metadata -> { documentIndex: 0, chunkIndex: i, text: '...' }
```

`embedMany` types its result without `metadata`; it is present at runtime, so
read it with a cast to `Embedding` from `genkit/embedder`.

## Multimodal embeddings

`voyage-multimodal-3.5` (text, images and mp4 video; 1024 dimensions) and
`voyage-multimodal-3` (text and images; 1024 dimensions). A document's text and
media parts become one interleaved input and yield one embedding. Images are
png, jpeg, webp or gif. Media is an `http(s)` URL or a base64 `data:` URL
(read a local file into a data URL to embed it). One request must use only
URLs or only base64 media.

```ts
import { readFileSync } from 'node:fs';
import { Document } from 'genkit';

const png = readFileSync('chart.png').toString('base64');
const [embedding] = await ai.embedMany({
  embedder: voyageai.embedder('voyage-multimodal-3.5'),
  content: [
    Document.fromParts([
      { text: 'Quarterly revenue' },
      {
        media: {
          url: `data:image/png;base64,${png}`,
          contentType: 'image/png',
        },
      },
    ]),
  ],
  options: { inputType: 'document' },
});
```

Options: `inputType` and `truncation`. `outputDimension` is not available: the
official SDK does not send it for the multimodal endpoint.

## Rerankers

`rerank-3`, `rerank-3-lite`, `rerank-2.5` and `rerank-2.5-lite`.

```ts
const ranked = await ai.rerank({
  reranker: voyageai.reranker('rerank-3'),
  query: 'flying animals',
  documents: [
    { content: [{ text: 'cats' }] },
    { content: [{ text: 'birds' }] },
  ],
  options: { topK: 1 },
});
ranked[0].score();
```

Options: `topK` and `truncation`. Documents are text only.

## Batching

A single `embed`, `embedMany` or contextualized call can hold any number of
inputs. The plugin splits it, in order, into several Voyage requests that each
respect the limits: at most 1000 inputs and the documented per-request token
ceiling of the model (120,000, 320,000 or 1,000,000 tokens; `voyage-code-4` has
no documented ceiling, so it is split by input count only). The results are
concatenated in input order, and `documentIndex` in contextualized metadata
counts across the whole call.

Tokens are counted with Voyage's published tokenizer for the model, downloaded
from Hugging Face on first use. It is only needed when an input is large enough
to come near a limit: a text of N bytes has at most N + 2 tokens, and when those
bounds fit one request nothing is tokenized and no download happens. To count
offline or differently, pass `countTokens: async (model, texts) => number[]` to
`voyageai()`. If the tokenizer cannot be loaded, the call fails with an error
instead of guessing.

- A contextualized document that alone exceeds the request ceiling is rejected
  (documents are never truncated silently).
- Inputs with images or video are charged the largest cost one input can have
  (32,000 tokens), because their token count depends on pixels that are not
  known client side. Such requests are therefore split conservatively.
- Rerank requests are not split: they take at most 1000 documents.

## Errors

Voyage error bodies look like `{"detail": "<message>"}`. The plugin raises a
`GenkitError` whose message carries that text, with a status mapped from the
HTTP status.

## Tests

```bash
pnpm test         # unit tests with a fake HTTP layer
VOYAGE_API_KEY=... pnpm test:live
```

License: Apache 2.0

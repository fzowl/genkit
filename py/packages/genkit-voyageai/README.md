# Genkit Voyage AI Plugin

[Voyage AI](https://www.voyageai.com/) embedders for Genkit Python: text,
contextualized chunk, and multimodal embeddings, through the official
[`voyageai`](https://pypi.org/project/voyageai/) SDK.

## Installation

```bash
uv add genkit genkit-voyageai
```

Get an API key from the [Voyage dashboard](https://dash.voyageai.com) and pass it
as `VoyageAI(api_key=...)` or set `VOYAGE_API_KEY`. A missing key fails when the
plugin initializes.

## Usage

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

Embedders are named `voyageai/<model>`, and `'voyageai/voyage-4'` works anywhere
`VoyageAI.embedder('voyage-4')` does. **There is no default model**: every call
names one, and a name outside the catalog below does not resolve (there is no
fallback).

### Plugin options

| Option     | Default                    | Meaning                                         |
| ---------- | -------------------------- | ----------------------------------------------- |
| `api_key`  | `VOYAGE_API_KEY`           | Voyage API key.                                 |
| `base_url` | `https://api.voyageai.com` | API host; the SDK's `/v1` is added for you.     |
| `timeout`  | SDK default                | Per-request timeout in seconds.                 |

## Embedders

| Kind           | Models                                                                                                                                                      | Dimensions (first is the default)                         |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Text           | `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-4`, `voyage-3-large`, `voyage-3.5`, `voyage-3.5-lite`, `voyage-code-3`                           | 1024, 2048, 512, 256                                      |
| Text           | `voyage-finance-2`, `voyage-law-2`                                                                                                                          | 1024 only                                                 |
| Contextualized | `voyage-context-4`, `voyage-context-3`                                                                                                                      | 1024, 2048, 512, 256                                      |
| Multimodal     | `voyage-multimodal-3.5` (text, image, video)                                                                                                                | 1024, 256, 512, 2048                                      |
| Multimodal     | `voyage-multimodal-3` (text, image)                                                                                                                         | 1024 only                                                 |

### Batching

One request carries at most 1000 inputs and, where Voyage documents a ceiling, a
number of tokens (120000 for `voyage-context-*`, 320000 for `voyage-multimodal-*`,
and the per-model figure for the text models; `voyage-code-4` has none documented,
so it is split by count only). A call that exceeds either is split automatically
into several requests, sent one after another, and the embeddings come back in
input order. For contextualized embedders `documentIndex` counts across the whole
call, not within one request.

Tokens are counted with the SDK's tokenizer (`client.tokenize`), but only when the
count can matter. The tokenizers normalize text to NFC and then split it into
byte-level tokens, so every token covers at least one UTF-8 byte of the normalized
text; when that byte length plus two already fits the ceiling for every input,
nothing is tokenized. Otherwise the
tokenizer runs in a worker thread and may download itself from the Hugging Face
Hub on first use. If it cannot load, the call fails with `UNAVAILABLE` naming the
cause rather than guessing a split. The count is the tokenizer's exact count,
which in our checks was never below the server's.

- Text and multimodal text: an input costs its token count, capped at the model's
  context length while `truncation` is on (the server truncates), uncapped when it
  is off.
- Multimodal media: image and video cost cannot be known from text (they are priced
  by pixels, and a URL's size is not visible), so an input with any media is
  counted at the per-input maximum, 32000 tokens.
- Contextualized: documents are chunked by Voyage, so each costs all its tokens.
  A single document larger than the request ceiling fits no request and raises
  `INVALID_ARGUMENT` naming its index, token count and the limit; nothing is
  truncated silently.

### Config

Pass these to `ai.embed(config=...)` as a dict (snake_case or camelCase) or as
`VoyageEmbedConfig` / `VoyageContextualizedEmbedConfig` to `VoyageAI.embedder(name, config=...)`.

| Field              | Applies to                  | Meaning                                                                                                          |
| ------------------ | --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `input_type`       | all (required for contextualized) | `'document'` or `'query'`. Left out of the request when unset, except for contextualized models.          |
| `output_dimension` | all                         | Vector width. Checked against the model's dimensions before the call; any other value is rejected.               |
| `truncation`       | text, multimodal            | Truncate over-long input instead of failing. Voyage defaults to `true`.                                          |

### Contextualized embedders

`voyage-context-*` embed each chunk of a document with the rest of the document
as context. `input_type` is required:

- `'document'`: every document is sent with `enable_auto_chunking` and
  `chunk_size=32000`, so Voyage splits it. Each document must be exactly one
  text part; a document that is already split into several parts is rejected.
- `'query'`: each document is one query and is never chunked.

The result is **one Genkit embedding per chunk**, flattened in (document, chunk)
order. Each embedding's `metadata` has `documentIndex`, `chunkIndex` and, when
Voyage returned it (document mode), the chunk `text`:

```python
chunks = await ai.embed(
    embedder=VoyageAI.embedder('voyage-context-4'),
    content=long_report_text,
    config={'input_type': 'document'},
)
for chunk in chunks:
    print(chunk.metadata['chunkIndex'], chunk.metadata['text'][:40])
```

### Multimodal embedders

A document's text and media parts, in order, become one input and one vector.
Media is an `http(s)` URL or a base64 `data:` URL of `image/png`, `image/jpeg`,
`image/webp` or `image/gif` (and `video/mp4` for `voyage-multimodal-3.5`).
Because Voyage does not accept both forms in one request, every media part in a
call must be a URL, or every one must be base64.

```python
from genkit import Document, Part

document = Document(content=[Part.from_text('a red square'), Part.from_media(data_url, 'image/png')])
vectors = await ai.embed(
    embedder=VoyageAI.embedder('voyage-multimodal-3.5'),
    content=document,
    config={'output_dimension': 256},
)
```

## Errors

A failed call raises `GenkitError` carrying the `detail` Voyage sent in its error
body, with the status mapped as usual (401 `UNAUTHENTICATED`, 429
`RESOURCE_EXHAUSTED`, 5xx `INTERNAL` or `UNAVAILABLE`), so Genkit's retry and
fallback middleware can act on it. Input Voyage would reject (unknown option,
unsupported dimension, wrong media) raises `INVALID_ARGUMENT` before any request.

Genkit Python has no reranker abstraction, so Voyage's rerank models are not part
of this plugin.

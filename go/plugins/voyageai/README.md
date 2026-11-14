# Voyage AI Plugin

The Voyage AI plugin provides Genkit embedders for the
[Voyage AI](https://docs.voyageai.com) API: text, contextualized chunk, and
multimodal embeddings. Voyage has no official Go SDK, so the plugin talks to the
REST API directly.

The plugin provides embedders only. Genkit Go has no reranker abstraction, so
Voyage's rerankers are not part of it.

## Setup

```bash
go get github.com/firebase/genkit/go/plugins/voyageai
```

Create an API key at <https://dash.voyageai.com> and set it:

```bash
export VOYAGE_API_KEY=<your-api-key>
```

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/voyageai"
)

ctx := context.Background()
g := genkit.Init(ctx, genkit.WithPlugins(&voyageai.VoyageAI{}))

res, err := genkit.Embed(ctx, g,
    ai.WithEmbedderName("voyageai/voyage-4"),
    ai.WithConfig(&voyageai.TextEmbedConfig{InputType: voyageai.InputTypeDocument}),
    ai.WithTextDocs("The cat sat on the mat."),
)
```

`APIKey` overrides the environment variable, and `BaseURL`, `Headers`, and
`HTTPClient` adjust how requests are sent. `Init` panics when no key is
configured.

## Models

Init registers one embedder per catalog model, named `voyageai/<model>`. There
is no default model: always name the embedder, and a name outside the catalog
does not resolve.

| Family | Models | Config |
|--------|--------|--------|
| Text | `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-4`, `voyage-3-large`, `voyage-3.5`, `voyage-3.5-lite`, `voyage-code-3`, `voyage-finance-2`, `voyage-law-2` | `TextEmbedConfig` |
| Contextualized | `voyage-context-4`, `voyage-context-3` | `ContextualizedEmbedConfig` |
| Multimodal | `voyage-multimodal-3.5` (text, image, video), `voyage-multimodal-3` (text, image) | `MultimodalEmbedConfig` |

Each config takes:

- `InputType`: `"document"` or `"query"`. Voyage prepends a retrieval prompt for
  the type. It is optional for text and multimodal embedders and required for
  contextualized ones.
- `OutputDimension`: the vector size. Models offer 1024 (the default), 2048,
  512, and 256; `voyage-finance-2`, `voyage-law-2`, and `voyage-multimodal-3`
  offer only 1024. Other values are rejected before a request is sent.
- `Truncation` (text and multimodal): whether over-long inputs are truncated
  (the API default) or rejected.

A request takes at most 1000 documents. Token limits are enforced by the
server, and its error message is returned as the error text.

## Contextualized embeddings

A contextualized embedder embeds each chunk of a document with the context of
the whole document. Send one document per `ai.Document`, with exactly one text
part, and `InputType: "document"`: Voyage chunks the document itself (chunks of
up to 32000 tokens). The response holds one embedding per chunk, flattened in
(document, chunk) order, and each embedding's `Metadata` carries
`documentIndex`, `chunkIndex`, and the chunk `text`.

```go
res, err := genkit.Embed(ctx, g,
    ai.WithEmbedderName("voyageai/voyage-context-4"),
    ai.WithConfig(&voyageai.ContextualizedEmbedConfig{InputType: voyageai.InputTypeDocument}),
    ai.WithTextDocs(longDocument),
)
for _, e := range res.Embeddings {
    fmt.Println(e.Metadata["documentIndex"], e.Metadata["chunkIndex"], e.Metadata["text"])
}
```

Queries use `InputType: "query"`: one embedding per query, with no chunking and
no `text` in the metadata. Pre-chunked documents (several text parts) are not
supported.

## Multimodal embeddings

A document becomes one input made of its text and media parts, in order. Media
is given as a base64 `data:` URL or an `http(s)` URL (with `ContentType` set).
Images may be PNG, JPEG, WebP, or GIF; video, for `voyage-multimodal-3.5` only,
is MP4. One request must give all its media in the same form, URLs or base64.

```go
res, err := genkit.Embed(ctx, g,
    ai.WithEmbedderName("voyageai/voyage-multimodal-3.5"),
    ai.WithDocs(&ai.Document{Content: []*ai.Part{
        ai.NewTextPart("A photo of a cat"),
        ai.NewMediaPart("image/png", "data:image/png;base64,"+encodedPNG),
    }}),
)
```

## Testing

Unit tests run against a local stub server. The live test makes a handful of
small API calls and needs a key:

```bash
GENKIT_LIVE=1 go test -run Live ./plugins/voyageai/
```

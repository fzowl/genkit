// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

// This sample embeds text with Voyage AI and ranks documents against a query.
//
// It shows the three things that set Voyage embedders apart:
//
//   - The model is always named. There is no default, so the flows below pick
//     voyage-4-lite and voyage-context-4 themselves.
//   - Documents and queries are embedded differently. Setting InputType to
//     "document" or "query" lets Voyage prepend the retrieval prompt each side
//     was trained with.
//   - Contextualized embedders (voyage-context-4) embed a whole document: Voyage
//     splits it into chunks and returns one embedding per chunk, with the chunk
//     text and its position in the embedding's metadata.
//
// Set VOYAGE_API_KEY, then run it:
//
//	go run .
//
// Or with the Dev UI, to call the flows from a browser:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP:
//
//	curl -X POST http://localhost:8080/searchFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"query": "Which animal purrs?"}}'
//
//	curl -X POST http://localhost:8080/chunksFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"document": "The cat sat on the mat. It purred."}}'
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"net/http"
	"sort"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/server"
	"github.com/firebase/genkit/go/plugins/voyageai"
)

// defaultDocuments is what searchFlow searches when the request names none.
var defaultDocuments = []string{
	"The cat sat on the mat and purred.",
	"Stock markets fell sharply on Tuesday.",
	"A recipe for a quick tomato soup.",
	"The kitten curled up on the rug.",
}

// SearchInput is the request of searchFlow. A field without omitempty is
// required; the defaults pre-fill the Dev UI form.
type SearchInput struct {
	Query     string   `json:"query" jsonschema:"default=Which animal purrs?"`
	Documents []string `json:"documents,omitempty" jsonschema_description:"Texts to rank; a built-in set is used when empty"`
}

// Hit is one ranked document.
type Hit struct {
	Document string  `json:"document"`
	Score    float64 `json:"score"`
}

// ChunksInput is the request of chunksFlow.
type ChunksInput struct {
	Document string `json:"document" jsonschema:"default=The cat sat on the mat. It was a sunny day, and the cat purred."`
}

// Chunk is one chunk Voyage cut from a document.
type Chunk struct {
	Text       string `json:"text"`
	Dimensions int    `json:"dimensions"`
}

func main() {
	ctx := context.Background()

	// The plugin reads the API key from VOYAGE_API_KEY.
	g := genkit.Init(ctx, genkit.WithPlugins(&voyageai.VoyageAI{}))

	// searchFlow embeds the documents as documents and the query as a query,
	// then ranks the documents by cosine similarity.
	genkit.DefineFlow(g, "searchFlow", func(ctx context.Context, in SearchInput) ([]Hit, error) {
		docs := in.Documents
		if len(docs) == 0 {
			docs = defaultDocuments
		}
		const embedder = "voyageai/voyage-4-lite"
		docRes, err := genkit.Embed(ctx, g,
			ai.WithEmbedderName(embedder),
			ai.WithConfig(&voyageai.TextEmbedConfig{InputType: voyageai.InputTypeDocument}),
			ai.WithTextDocs(docs...))
		if err != nil {
			return nil, fmt.Errorf("could not embed documents: %w", err)
		}
		queryRes, err := genkit.Embed(ctx, g,
			ai.WithEmbedderName(embedder),
			ai.WithConfig(&voyageai.TextEmbedConfig{InputType: voyageai.InputTypeQuery}),
			ai.WithTextDocs(in.Query))
		if err != nil {
			return nil, fmt.Errorf("could not embed query: %w", err)
		}

		hits := make([]Hit, len(docs))
		for i, e := range docRes.Embeddings {
			hits[i] = Hit{Document: docs[i], Score: cosine(queryRes.Embeddings[0].Embedding, e.Embedding)}
		}
		sort.Slice(hits, func(a, b int) bool { return hits[a].Score > hits[b].Score })
		return hits, nil
	})

	// chunksFlow embeds one document with a contextualized embedder, which
	// chunks it server-side and returns one embedding per chunk.
	genkit.DefineFlow(g, "chunksFlow", func(ctx context.Context, in ChunksInput) ([]Chunk, error) {
		res, err := genkit.Embed(ctx, g,
			ai.WithEmbedderName("voyageai/voyage-context-4"),
			ai.WithConfig(&voyageai.ContextualizedEmbedConfig{InputType: voyageai.InputTypeDocument}),
			ai.WithTextDocs(in.Document))
		if err != nil {
			return nil, fmt.Errorf("could not embed document: %w", err)
		}
		chunks := make([]Chunk, len(res.Embeddings))
		for i, e := range res.Embeddings {
			text, _ := e.Metadata["text"].(string)
			chunks[i] = Chunk{Text: text, Dimensions: len(e.Embedding)}
		}
		return chunks, nil
	})

	// Serve every flow over HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	if err := server.Start(ctx, "127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}

// cosine is the cosine similarity of a and b.
func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

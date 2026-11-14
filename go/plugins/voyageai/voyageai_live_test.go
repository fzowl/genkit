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

package voyageai_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/firebase/genkit/go/plugins/voyageai"
)

// The live test makes a handful of tiny calls. It needs GENKIT_LIVE=1 and
// VOYAGE_API_KEY.
func TestPluginLive(t *testing.T) {
	livetest.Env(t, "VOYAGE_API_KEY")
	g := livetest.Init(t, &voyageai.VoyageAI{})
	ctx := t.Context()

	t.Run("text", func(t *testing.T) {
		livetest.RunEmbedder(t, g, livetest.EmbedderSuite{
			Embedder:   ai.NewEmbedderRef("voyageai/voyage-4-lite", &voyageai.TextEmbedConfig{InputType: voyageai.InputTypeDocument, OutputDimension: 256}),
			Dimensions: 256,
			Normalized: true,
		})
		res, err := genkit.Embed(ctx, g,
			ai.WithEmbedderName("voyageai/voyage-4-lite"),
			ai.WithConfig(&voyageai.TextEmbedConfig{InputType: voyageai.InputTypeQuery, OutputDimension: 256}),
			ai.WithTextDocs("where did the cat sit?"))
		if err != nil {
			t.Fatalf("Embed() error = %v", err)
		}
		if len(res.Embeddings) != 1 || len(res.Embeddings[0].Embedding) != 256 {
			t.Errorf("query embedding = %d vectors, want 1 of 256 dimensions", len(res.Embeddings))
		}
	})

	t.Run("contextualized", func(t *testing.T) {
		const doc = "The cat sat on the mat. It was a sunny day, and the cat purred."
		res, err := genkit.Embed(ctx, g,
			ai.WithEmbedderName("voyageai/voyage-context-4"),
			ai.WithConfig(&voyageai.ContextualizedEmbedConfig{InputType: voyageai.InputTypeDocument, OutputDimension: 256}),
			ai.WithTextDocs(doc))
		if err != nil {
			t.Fatalf("Embed(document) error = %v", err)
		}
		if len(res.Embeddings) == 0 {
			t.Fatal("Embed(document) returned no embeddings")
		}
		var chunks []string
		for i, e := range res.Embeddings {
			text, _ := e.Metadata["text"].(string)
			if len(e.Embedding) != 256 || text == "" || e.Metadata["documentIndex"] != 0 || e.Metadata["chunkIndex"] != i {
				t.Errorf("embedding %d: %d dimensions, metadata %v, want 256 dimensions and document 0, chunk %d, with text", i, len(e.Embedding), e.Metadata, i)
			}
			chunks = append(chunks, text)
		}
		if got := strings.Join(chunks, ""); !strings.Contains(got, "cat purred") {
			t.Errorf("chunk texts = %q, want the document's text", chunks)
		}

		res, err = genkit.Embed(ctx, g,
			ai.WithEmbedderName("voyageai/voyage-context-4"),
			ai.WithConfig(&voyageai.ContextualizedEmbedConfig{InputType: voyageai.InputTypeQuery, OutputDimension: 256}),
			ai.WithTextDocs("where did the cat sit?"))
		if err != nil {
			t.Fatalf("Embed(query) error = %v", err)
		}
		if len(res.Embeddings) != 1 || len(res.Embeddings[0].Embedding) != 256 {
			t.Errorf("query embedding = %d vectors, want 1 of 256 dimensions", len(res.Embeddings))
		}
		if _, ok := res.Embeddings[0].Metadata["text"]; ok {
			t.Error("query embedding has text metadata, want none")
		}
	})

	t.Run("multimodal", func(t *testing.T) {
		img := image.NewRGBA(image.Rect(0, 0, 64, 64))
		for x := range 64 {
			for y := range 64 {
				img.Set(x, y, color.RGBA{R: uint8(4 * x), G: uint8(4 * y), B: 128, A: 255})
			}
		}
		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
			t.Fatal(err)
		}
		dataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
		res, err := genkit.Embed(ctx, g,
			ai.WithEmbedderName("voyageai/voyage-multimodal-3.5"),
			ai.WithConfig(&voyageai.MultimodalEmbedConfig{InputType: voyageai.InputTypeDocument, OutputDimension: 256}),
			ai.WithDocs(&ai.Document{Content: []*ai.Part{ai.NewTextPart("a color gradient"), ai.NewMediaPart("image/png", dataURL)}}))
		if err != nil {
			t.Fatalf("Embed() error = %v", err)
		}
		if len(res.Embeddings) != 1 || len(res.Embeddings[0].Embedding) != 256 {
			t.Errorf("embedding = %d vectors, want 1 of 256 dimensions", len(res.Embeddings))
		}
	})

	t.Run("bad key", func(t *testing.T) {
		g := genkit.Init(context.Background(), genkit.WithPlugins(&voyageai.VoyageAI{APIKey: "invalid"}))
		_, err := genkit.Embed(ctx, g, ai.WithEmbedderName("voyageai/voyage-4-lite"), ai.WithTextDocs("x"))
		if err == nil || !strings.Contains(err.Error(), "HTTP 401") {
			t.Errorf("Embed() error = %v, want an HTTP 401 error carrying Voyage's detail", err)
		}
		t.Logf("bad key error: %v", err)
	})
}

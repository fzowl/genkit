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
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/voyageai"
)

// fakeAPI is a Voyage server stub. It records the last decoded request body
// and how many requests arrived.
type fakeAPI struct {
	hits atomic.Int32
	mu   sync.Mutex
	body map[string]any
}

// lastBody returns the body of the most recent request.
func (a *fakeAPI) lastBody() map[string]any {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.body
}

// newGenkit starts a Genkit app whose plugin talks to a server answering
// every request with handler.
func newGenkit(t *testing.T, handler func(w http.ResponseWriter, r *http.Request)) (*genkit.Genkit, *fakeAPI) {
	t.Helper()
	api := &fakeAPI{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		api.hits.Add(1)
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want the configured bearer key", got)
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		api.mu.Lock()
		api.body = body
		api.mu.Unlock()
		handler(w, r)
	}))
	t.Cleanup(srv.Close)
	g := genkit.Init(context.Background(), genkit.WithPlugins(&voyageai.VoyageAI{APIKey: "test-key", BaseURL: srv.URL}))
	return g, api
}

func reply(status int, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}
}

func embed(g *genkit.Genkit, model string, cfg any, docs ...*ai.Document) (*ai.EmbedResponse, error) {
	opts := []ai.EmbedderOption{ai.WithEmbedderName("voyageai/" + model), ai.WithDocs(docs...)}
	if cfg != nil {
		opts = append(opts, ai.WithConfig(cfg))
	}
	return genkit.Embed(context.Background(), g, opts...)
}

func text(s ...string) []*ai.Document {
	docs := make([]*ai.Document, len(s))
	for i, t := range s {
		docs[i] = ai.DocumentFromText(t, nil)
	}
	return docs
}

func TestInitWithoutKeyPanics(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "")
	defer func() {
		r := recover()
		if r == nil || !strings.Contains(r.(string), "VOYAGE_API_KEY") {
			t.Errorf("Init panic = %v, want a message naming VOYAGE_API_KEY", r)
		}
	}()
	genkit.Init(context.Background(), genkit.WithPlugins(&voyageai.VoyageAI{}))
}

func TestUnknownModel(t *testing.T) {
	g, api := newGenkit(t, reply(200, "{}"))
	if voyageai.Embedder(g, "voyage-9000") != nil {
		t.Error("Embedder(voyage-9000) is not nil, want no embedder outside the catalog")
	}
	if _, err := embed(g, "voyage-9000", nil, text("x")...); err == nil || !strings.Contains(err.Error(), "voyageai/voyage-9000") {
		t.Errorf("Embed with an unknown model error = %v, want one naming voyageai/voyage-9000", err)
	}
	if n := api.hits.Load(); n != 0 {
		t.Errorf("server received %d requests, want 0", n)
	}
}

// TestRejectedBeforeSending covers every request the plugin refuses itself:
// the server must never see them.
func TestRejectedBeforeSending(t *testing.T) {
	image := ai.NewMediaPart("image/png", "data:image/png;base64,AAAA")
	imageURL := ai.NewMediaPart("image/png", "https://example.com/a.png")
	b64Video := ai.NewMediaPart("video/mp4", "data:video/mp4;base64,AAAA")

	tests := []struct {
		name    string
		model   string
		cfg     any
		docs    []*ai.Document
		wantErr string
	}{
		{"text/too many inputs", "voyage-4", nil, text(make([]string, 1001)...), "1001 input documents exceed the limit of 1000"},
		{"text/bad input type", "voyage-4", &voyageai.TextEmbedConfig{InputType: "passage"}, text("x"), "inputType"},
		{"text/dimension not offered", "voyage-4", &voyageai.TextEmbedConfig{OutputDimension: 100}, text("x"), "outputDimension 100"},
		{"text/fixed dimension model", "voyage-law-2", &voyageai.TextEmbedConfig{OutputDimension: 512}, text("x"), "outputDimension 512"},
		{"text/media", "voyage-4", nil, []*ai.Document{{Content: []*ai.Part{ai.NewTextPart("x"), image}}}, "only text parts"},
		{"contextualized/missing input type", "voyage-context-4", nil, text("x"), "inputType is required"},
		{"contextualized/bad dimension", "voyage-context-4", &voyageai.ContextualizedEmbedConfig{InputType: "query", OutputDimension: 4096}, text("x"), "outputDimension 4096"},
		{"contextualized/pre-chunked document", "voyage-context-4", &voyageai.ContextualizedEmbedConfig{InputType: "document"}, []*ai.Document{{Content: []*ai.Part{ai.NewTextPart("a"), ai.NewTextPart("b")}}}, "exactly one text part"},
		{"contextualized/media", "voyage-context-3", &voyageai.ContextualizedEmbedConfig{InputType: "document"}, []*ai.Document{{Content: []*ai.Part{image}}}, "exactly one text part"},
		{"multimodal/bad dimension", "voyage-multimodal-3", &voyageai.MultimodalEmbedConfig{OutputDimension: 256}, text("x"), "outputDimension 256"},
		{"multimodal/mixed url and base64", "voyage-multimodal-3.5", nil, []*ai.Document{{Content: []*ai.Part{imageURL}}, {Content: []*ai.Part{image}}}, "cannot mix"},
		{"multimodal/mixed within a document", "voyage-multimodal-3.5", nil, []*ai.Document{{Content: []*ai.Part{b64Video, imageURL}}}, "cannot mix"},
		{"multimodal/video on multimodal-3", "voyage-multimodal-3", nil, []*ai.Document{{Content: []*ai.Part{b64Video}}}, "does not support video"},
		{"multimodal/unsupported media type", "voyage-multimodal-3.5", nil, []*ai.Document{{Content: []*ai.Part{ai.NewMediaPart("image/bmp", "data:image/bmp;base64,AAAA")}}}, "image/bmp"},
		{"multimodal/audio", "voyage-multimodal-3.5", nil, []*ai.Document{{Content: []*ai.Part{ai.NewMediaPart("audio/mpeg", "https://example.com/a.mp3")}}}, "audio/mpeg"},
		{"multimodal/content type disagrees with data url", "voyage-multimodal-3.5", nil, []*ai.Document{{Content: []*ai.Part{ai.NewMediaPart("image/jpeg", "data:image/png;base64,AAAA")}}}, "does not match"},
		{"multimodal/url without content type", "voyage-multimodal-3.5", nil, []*ai.Document{{Content: []*ai.Part{ai.NewMediaPart("", "https://example.com/a")}}}, "contentType is required"},
		{"multimodal/non-base64 data url", "voyage-multimodal-3.5", nil, []*ai.Document{{Content: []*ai.Part{ai.NewMediaPart("image/png", "data:image/png,raw")}}}, "base64"},
		{"multimodal/empty document", "voyage-multimodal-3.5", nil, []*ai.Document{{Content: []*ai.Part{ai.NewTextPart("")}}}, "no text or media"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			g, api := newGenkit(t, reply(200, "{}"))
			_, err := embed(g, tc.model, tc.cfg, tc.docs...)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Embed error = %v, want one containing %q", err, tc.wantErr)
			}
			if !errors.Is(err, status.ErrInvalidArgument) {
				t.Errorf("Embed error status = %v, want INVALID_ARGUMENT", err)
			}
			if n := api.hits.Load(); n != 0 {
				t.Errorf("server received %d requests, want 0", n)
			}
		})
	}
}

func TestAPIErrorDetail(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   string
		is     *status.Sentinel
	}{
		{"detail string", 400, `{"detail":"Request to model 'voyage-4' failed. The max allowed tokens per submitted batch is 320000."}`, "max allowed tokens per submitted batch is 320000", status.ErrInvalidArgument},
		{"unauthorized", 401, `{"detail":"Provided API key is invalid."}`, "Provided API key is invalid.", status.ErrUnauthenticated},
		{"rate limited", 429, `{"detail":"You have exceeded your rate limit."}`, "exceeded your rate limit", status.ErrResourceExhausted},
		{"structured detail", 422, `{"detail":[{"loc":["body","input"],"msg":"field required"}]}`, "field required", status.ErrInvalidArgument},
		{"not json", 503, "upstream unavailable", "upstream unavailable", status.ErrUnavailable},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			g, _ := newGenkit(t, reply(tc.status, tc.body))
			_, err := embed(g, "voyage-4", nil, text("x")...)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Embed error = %v, want one containing %q", err, tc.want)
			}
			if !errors.Is(err, tc.is) {
				t.Errorf("Embed error = %v, want status %v", err, tc.is)
			}
		})
	}
}

func TestMalformedResponse(t *testing.T) {
	g, _ := newGenkit(t, reply(200, `{"data":[{"embedding":[1],"index":0}]}`))
	if _, err := embed(g, "voyage-4", nil, text("a", "b")...); err == nil || !strings.Contains(err.Error(), "1 embeddings for 2 inputs") {
		t.Errorf("Embed error = %v, want a count mismatch", err)
	}
	g, _ = newGenkit(t, reply(200, `{"data":[{"embedding":[1],"index":0},{"embedding":[2],"index":0}]}`))
	if _, err := embed(g, "voyage-4", nil, text("a", "b")...); err == nil || !strings.Contains(err.Error(), "index") {
		t.Errorf("Embed error = %v, want an index error", err)
	}
}

// TestContextualizedWireShape pins the API constraints found live: document
// mode must auto-chunk over a flat list of strings, query mode must not.
func TestContextualizedWireShape(t *testing.T) {
	g, api := newGenkit(t, reply(200, `{"data":[{"index":0,"data":[{"embedding":[1],"index":0}]}]}`))

	if _, err := embed(g, "voyage-context-4", &voyageai.ContextualizedEmbedConfig{InputType: "document"}, text("doc")...); err != nil {
		t.Fatal(err)
	}
	body := api.lastBody()
	if body["enable_auto_chunking"] != true || body["chunk_size"] != float64(32000) || body["input_type"] != "document" {
		t.Errorf("document request = %v, want enable_auto_chunking true, chunk_size 32000, input_type document", body)
	}
	if _, overlap := body["chunk_overlap"]; overlap {
		t.Error("document request carries chunk_overlap, want none")
	}
	if in, ok := body["inputs"].([]any); !ok || len(in) != 1 || in[0] != "doc" {
		t.Errorf("document inputs = %v, want a flat list of strings", body["inputs"])
	}

	res, err := embed(g, "voyage-context-4", &voyageai.ContextualizedEmbedConfig{InputType: "query"}, text("q")...)
	if err != nil {
		t.Fatal(err)
	}
	body = api.lastBody()
	for _, k := range []string{"enable_auto_chunking", "chunk_size", "chunk_overlap"} {
		if _, ok := body[k]; ok {
			t.Errorf("query request carries %s, want none", k)
		}
	}
	if _, ok := res.Embeddings[0].Metadata["text"]; ok {
		t.Error("query embedding carries text metadata, want none when the API returned none")
	}
}

func TestContextualizedChunkMapping(t *testing.T) {
	// Results and chunks arrive out of order; the output is flattened in
	// (document, chunk) order.
	g, _ := newGenkit(t, reply(200, `{"data":[
		{"index":1,"data":[{"embedding":[3],"index":0,"text":"b0"}]},
		{"index":0,"data":[{"embedding":[2],"index":1,"text":"a1"},{"embedding":[1],"index":0,"text":"a0"}]}]}`))
	res, err := embed(g, "voyage-context-4", &voyageai.ContextualizedEmbedConfig{InputType: "document"}, text("a", "b")...)
	if err != nil {
		t.Fatal(err)
	}
	want := []struct {
		vec        float32
		doc, chunk int
		text       string
	}{{1, 0, 0, "a0"}, {2, 0, 1, "a1"}, {3, 1, 0, "b0"}}
	if len(res.Embeddings) != len(want) {
		t.Fatalf("got %d embeddings, want %d", len(res.Embeddings), len(want))
	}
	for i, w := range want {
		e := res.Embeddings[i]
		m := e.Metadata
		if e.Embedding[0] != w.vec || m["documentIndex"] != w.doc || m["chunkIndex"] != w.chunk || m["text"] != w.text {
			t.Errorf("embedding %d = %v %v, want vector %v, document %d, chunk %d, text %q", i, e.Embedding, m, w.vec, w.doc, w.chunk, w.text)
		}
	}
}

func TestEmptyInputNeedsNoRequest(t *testing.T) {
	g, api := newGenkit(t, reply(200, "{}"))
	for _, model := range []string{"voyage-4", "voyage-context-4", "voyage-multimodal-3.5"} {
		// No config at all (which voyage-context-4 would normally reject): there is nothing to embed.
		res, err := embed(g, model, nil)
		if err != nil || res == nil || len(res.Embeddings) != 0 {
			t.Errorf("%s: Embed(no documents) = %v, %v, want an empty response", model, res, err)
		}
	}
	if n := api.hits.Load(); n != 0 {
		t.Errorf("server received %d requests, want 0", n)
	}
}

// TestIncompleteResponsesFailClosed: a vector or result the API left empty or
// repeated must be an error, not a silently short result.
func TestIncompleteResponsesFailClosed(t *testing.T) {
	ctxCfg := &voyageai.ContextualizedEmbedConfig{InputType: "document"}
	tests := []struct {
		name, model, body string
		cfg               any
		want              string
	}{
		{"text/empty vector", "voyage-4", `{"data":[{"embedding":[],"index":0}]}`, nil, "empty embedding"},
		{"multimodal/empty vector", "voyage-multimodal-3.5", `{"data":[{"index":0}]}`, nil, "empty embedding"},
		{"contextualized/empty data", "voyage-context-4", `{"data":[{"index":0,"data":[]}]}`, ctxCfg, "no embeddings for input 0"},
		{"contextualized/empty vector", "voyage-context-4", `{"data":[{"index":0,"data":[{"embedding":[],"index":0}]}]}`, ctxCfg, "empty embedding"},
		{"contextualized/repeated chunk index", "voyage-context-4", `{"data":[{"index":0,"data":[{"embedding":[1],"index":0},{"embedding":[2],"index":0}]}]}`, ctxCfg, "chunk index"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			g, _ := newGenkit(t, reply(200, tc.body))
			if _, err := embed(g, tc.model, tc.cfg, text("x")...); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("Embed error = %v, want one containing %q", err, tc.want)
			}
		})
	}
}

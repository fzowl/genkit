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

// Package voyageai is a Genkit plugin for the Voyage AI embedding API
// (https://docs.voyageai.com).
//
// The plugin registers one embedder per model of Voyage's embedding catalog,
// named voyageai/<model>, in three families:
//
//   - text embedders, such as voyageai/voyage-4 (POST /v1/embeddings),
//     configured with [TextEmbedConfig];
//   - contextualized chunk embedders, voyageai/voyage-context-4 and
//     voyageai/voyage-context-3 (POST /v1/contextualizedembeddings),
//     configured with [ContextualizedEmbedConfig];
//   - multimodal embedders, voyageai/voyage-multimodal-3.5 and
//     voyageai/voyage-multimodal-3 (POST /v1/multimodalembeddings),
//     configured with [MultimodalEmbedConfig].
//
// There is no default model: an embedder is always named explicitly, and a
// name outside the catalog does not resolve.
//
// Voyage has no official Go SDK, so the plugin talks to the REST API directly.
package voyageai

import (
	"context"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal"
)

const provider = "voyageai"

// Values for the InputType field of the embedder configs.
const (
	// InputTypeDocument marks the inputs as documents to be searched.
	InputTypeDocument = "document"
	// InputTypeQuery marks the inputs as queries against documents.
	InputTypeQuery = "query"
)

// VoyageAI is a Genkit plugin for the Voyage AI embedding API.
type VoyageAI struct {
	// APIKey authenticates requests. When empty, the VOYAGE_API_KEY
	// environment variable is used, and [VoyageAI.Init] panics if that is
	// empty too.
	APIKey string
	// BaseURL overrides the API endpoint, https://api.voyageai.com by default.
	// Optional.
	BaseURL string
	// Headers are extra HTTP headers sent with every request. Optional.
	Headers http.Header
	// HTTPClient overrides the HTTP client used for API requests, to set
	// timeouts, proxies, or tracing. Defaults to http.DefaultClient. Optional.
	HTTPClient *http.Client

	mu      sync.Mutex
	initted bool
}

// Name returns the plugin's name, which prefixes the name of every embedder it
// registers.
func (v *VoyageAI) Name() string { return provider }

// Init registers an embedder for every model in the catalog. It panics when no
// API key is configured or when called twice, both setup mistakes rather than
// conditions an application can recover from.
func (v *VoyageAI) Init(ctx context.Context) []api.Action {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.initted {
		panic("voyageai: plugin already initialized")
	}

	apiKey := v.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("VOYAGE_API_KEY")
	}
	if apiKey == "" {
		panic("voyageai: an API key is required: set APIKey or the VOYAGE_API_KEY environment variable (keys are created at https://dash.voyageai.com)")
	}
	baseURL := v.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	httpClient := v.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	headers := make(http.Header, len(v.Headers))
	for k, vs := range v.Headers {
		headers[http.CanonicalHeaderKey(k)] = append([]string(nil), vs...)
	}
	c := &client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		httpClient: httpClient,
		headers:    headers,
	}

	actions := make([]api.Action, 0, len(models))
	for i := range models {
		actions = append(actions, newEmbedder(c, &models[i]))
	}
	v.initted = true
	return actions
}

// Embedder returns the embedder for the given catalog model, or nil if the
// model is not in the catalog or the plugin is not registered with g.
func Embedder(g *genkit.Genkit, name string) ai.Embedder {
	return genkit.LookupEmbedder(g, api.NewName(provider, internal.TrimProvider(provider, name)))
}

// newEmbedder builds the embedder action for m.
func newEmbedder(c *client, m *model) *ai.EmbedderAction {
	limits := map[string]any{
		"dimensions":    m.dimensions,
		"contextTokens": m.contextTokens,
	}
	if m.maxRequestTokens > 0 {
		limits["maxRequestTokens"] = m.maxRequestTokens
	}
	opts := &ai.EmbedderOptions{
		Label:      internal.ProviderLabel("Voyage AI", m.name),
		Dimensions: m.dimensions[0],
		Supports:   &ai.EmbedderSupports{Input: m.input},
		Metadata:   map[string]any{"voyageai": limits},
	}
	name := api.NewName(provider, m.name)
	switch m.kind {
	case kindContextualized:
		return ai.NewEmbedderAction(name, opts, func(ctx context.Context, req *ai.EmbedRequest, cfg ContextualizedEmbedConfig) (*ai.EmbedResponse, error) {
			return embedContextualized(ctx, c, m, req, cfg)
		})
	case kindMultimodal:
		return ai.NewEmbedderAction(name, opts, func(ctx context.Context, req *ai.EmbedRequest, cfg MultimodalEmbedConfig) (*ai.EmbedResponse, error) {
			return embedMultimodal(ctx, c, m, req, cfg)
		})
	default:
		return ai.NewEmbedderAction(name, opts, func(ctx context.Context, req *ai.EmbedRequest, cfg TextEmbedConfig) (*ai.EmbedResponse, error) {
			return embedText(ctx, c, m, req, cfg)
		})
	}
}

// checkInputType validates an input type. An empty one is allowed unless
// required.
func checkInputType(inputType string, required bool) error {
	switch inputType {
	case InputTypeDocument, InputTypeQuery:
		return nil
	case "":
		if required {
			return invalidArgument("inputType is required, want %q or %q", InputTypeDocument, InputTypeQuery)
		}
		return nil
	}
	return invalidArgument("inputType %q is not supported, want %q or %q", inputType, InputTypeDocument, InputTypeQuery)
}

// checkInputCount validates the number of inputs of a request. Callers answer
// an empty list themselves, before any validation, so n is at least one.
func checkInputCount(n int) error {
	if n > maxInputs {
		return invalidArgument("%d input documents exceed the limit of %d per request", n, maxInputs)
	}
	return nil
}

// invalidArgument reports a request the plugin rejects before sending it.
func invalidArgument(format string, args ...any) error {
	return status.Errorf(status.ErrInvalidArgument, "voyageai: "+format, args...)
}

// emptyResponse answers a request with no input documents, which needs no call.
func emptyResponse() *ai.EmbedResponse {
	return &ai.EmbedResponse{Embeddings: []*ai.Embedding{}}
}

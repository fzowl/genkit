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

package voyageai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal"
)

const (
	defaultBaseURL = "https://api.voyageai.com"

	// maxInputs is how many inputs one request may carry, on every endpoint.
	maxInputs = 1000

	// maxErrorBody bounds how much of an error response is read.
	maxErrorBody = 1 << 20
	// maxErrorText bounds how much of an unstructured error body is quoted.
	maxErrorText = 512
)

// client is a minimal REST client for the Voyage AI API. Voyage has no
// official Go SDK, so the plugin speaks the three embedding endpoints itself.
type client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	headers    http.Header
}

// post sends in as JSON to path and decodes the response into out. A non-2xx
// response becomes a [status.Error] carrying the text of Voyage's "detail"
// field.
func (c *client) post(ctx context.Context, path string, in, out any) error {
	body, err := json.Marshal(in)
	if err != nil {
		return fmt.Errorf("voyageai: encode request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("voyageai: create request: %w", err)
	}
	for k, v := range c.headers {
		req.Header[k] = append([]string(nil), v...)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("User-Agent", "genkit-go/"+internal.Version)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("voyageai: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return apiError(resp)
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("voyageai: decode response: %w", err)
	}
	return nil
}

// apiError turns an error response into a [status.Error] whose message is the
// "detail" Voyage sent, so the cause (an unsupported dimension, an over-long
// input, a bad key) reaches the caller instead of a bare status code.
func apiError(resp *http.Response) error {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBody))
	name := status.FromHTTPCode(resp.StatusCode)
	if resp.StatusCode == http.StatusUnprocessableEntity {
		// Request validation failures; the status table leaves 422 unmapped.
		name = status.InvalidArgument
	}
	return status.Errorf(status.Base(name),
		"voyageai: %s (HTTP %d)", errorDetail(raw, resp.Status), resp.StatusCode)
}

// errorDetail extracts the message from a Voyage error body, {"detail": "..."}.
// Validation failures can carry a structured detail instead of a string; that
// is rendered as its JSON. A body that is not in the documented shape is
// quoted, truncated, and fallback stands in for an empty one.
func errorDetail(body []byte, fallback string) string {
	var env struct {
		Detail json.RawMessage `json:"detail"`
	}
	if json.Unmarshal(body, &env) == nil && len(env.Detail) > 0 && string(env.Detail) != "null" {
		var s string
		if json.Unmarshal(env.Detail, &s) == nil {
			return s
		}
		return string(env.Detail)
	}
	text := strings.TrimSpace(string(body))
	if text == "" {
		return fallback
	}
	if len(text) > maxErrorText {
		text = text[:maxErrorText] + "..."
	}
	return text
}

// embeddingItem is one vector in a response, with the position of the input
// it belongs to.
type embeddingItem struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// orderByIndex returns the vectors of items ordered by index, checking that
// the response holds exactly one per input.
func orderByIndex(items []embeddingItem, want int) ([][]float32, error) {
	if len(items) != want {
		return nil, fmt.Errorf("voyageai: response has %d embeddings for %d inputs", len(items), want)
	}
	out := make([][]float32, want)
	seen := make([]bool, want)
	for _, it := range items {
		if it.Index < 0 || it.Index >= want || seen[it.Index] {
			return nil, fmt.Errorf("voyageai: response has invalid or repeated embedding index %d", it.Index)
		}
		if len(it.Embedding) == 0 {
			return nil, fmt.Errorf("voyageai: response has an empty embedding at index %d", it.Index)
		}
		seen[it.Index] = true
		out[it.Index] = it.Embedding
	}
	return out, nil
}

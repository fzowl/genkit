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

import "slices"

// kind is the Voyage endpoint family a model is served by.
type kind int

const (
	kindText kind = iota
	kindContextualized
	kindMultimodal
)

// model describes one entry of the Voyage embedding catalog. The plugin
// serves exactly these models and has no default among them.
type model struct {
	name string
	kind kind
	// dimensions lists the supported output dimensions; the first is the
	// model's default.
	dimensions []int
	// input lists the modalities the model accepts.
	input []string
	// contextTokens is the documented context window, per input (per chunk
	// for contextualized models). The server enforces it.
	contextTokens int
	// maxRequestTokens is the documented ceiling on the tokens of one
	// request, or 0 where Voyage does not document one. The server enforces
	// it.
	maxRequestTokens int
}

var (
	flexibleDimensions = []int{1024, 2048, 512, 256}
	fixedDimensions    = []int{1024}
	textOnly           = []string{"text"}
	textAndImage       = []string{"text", "image"}
	textImageVideo     = []string{"text", "image", "video"}
)

// models is the full catalog, in the order the models are registered.
var models = []model{
	{name: "voyage-4-large", kind: kindText, dimensions: flexibleDimensions, input: textOnly, contextTokens: 32000, maxRequestTokens: 120000},
	{name: "voyage-4", kind: kindText, dimensions: flexibleDimensions, input: textOnly, contextTokens: 32000, maxRequestTokens: 320000},
	{name: "voyage-4-lite", kind: kindText, dimensions: flexibleDimensions, input: textOnly, contextTokens: 32000, maxRequestTokens: 1000000},
	{name: "voyage-code-4", kind: kindText, dimensions: flexibleDimensions, input: textOnly, contextTokens: 32000},
	{name: "voyage-3-large", kind: kindText, dimensions: flexibleDimensions, input: textOnly, contextTokens: 32000, maxRequestTokens: 120000},
	{name: "voyage-3.5", kind: kindText, dimensions: flexibleDimensions, input: textOnly, contextTokens: 32000, maxRequestTokens: 320000},
	{name: "voyage-3.5-lite", kind: kindText, dimensions: flexibleDimensions, input: textOnly, contextTokens: 32000, maxRequestTokens: 1000000},
	{name: "voyage-code-3", kind: kindText, dimensions: flexibleDimensions, input: textOnly, contextTokens: 32000, maxRequestTokens: 120000},
	{name: "voyage-finance-2", kind: kindText, dimensions: fixedDimensions, input: textOnly, contextTokens: 32000, maxRequestTokens: 120000},
	{name: "voyage-law-2", kind: kindText, dimensions: fixedDimensions, input: textOnly, contextTokens: 16000, maxRequestTokens: 120000},

	{name: "voyage-context-4", kind: kindContextualized, dimensions: flexibleDimensions, input: textOnly, contextTokens: 32000, maxRequestTokens: 120000},
	{name: "voyage-context-3", kind: kindContextualized, dimensions: flexibleDimensions, input: textOnly, contextTokens: 32000, maxRequestTokens: 120000},

	{name: "voyage-multimodal-3.5", kind: kindMultimodal, dimensions: []int{1024, 256, 512, 2048}, input: textImageVideo, contextTokens: 32000, maxRequestTokens: 320000},
	{name: "voyage-multimodal-3", kind: kindMultimodal, dimensions: fixedDimensions, input: textAndImage, contextTokens: 32000, maxRequestTokens: 320000},
}

// validateDimension rejects an output dimension the model does not offer. Zero
// means the caller did not ask for one.
func (m *model) validateDimension(dim int) error {
	if dim == 0 || slices.Contains(m.dimensions, dim) {
		return nil
	}
	return invalidArgument("%s: outputDimension %d is not supported, want one of %v", m.name, dim, m.dimensions)
}

func (m *model) supports(modality string) bool {
	return slices.Contains(m.input, modality)
}

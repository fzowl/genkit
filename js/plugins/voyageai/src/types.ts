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
import type { TokenCounter } from './batching.js';

/** Options of the `voyageai` plugin. */
export interface VoyageAIPluginOptions {
  /** Voyage API key. Defaults to the `VOYAGE_API_KEY` environment variable. */
  apiKey?: string;
  /** Overrides the API base URL (defaults to `https://api.voyageai.com/v1`). */
  baseUrl?: string;
  /** Maximum number of retries of a failed request (SDK default applies when unset). */
  maxRetries?: number;
  /** Request timeout in seconds (SDK default applies when unset). */
  timeoutInSeconds?: number;
  /** Custom `fetch` implementation, e.g. for proxies or tests. */
  fetch?: typeof fetch;
  /**
   * Counts tokens for splitting large inputs into token-limited requests.
   * Defaults to Voyage's published tokenizers, downloaded from Hugging Face on
   * first use (only needed when an input is large enough to approach a request limit).
   */
  countTokens?: TokenCounter;
}

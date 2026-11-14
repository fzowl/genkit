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

/** Where Voyage publishes the tokenizer of each of its models. */
const TOKENIZER_URL = 'https://huggingface.co/voyageai';

type Tokenizer = { encode(text: string): { ids: number[] } };

async function fetchJson(fetchFn: typeof fetch, url: string): Promise<unknown> {
  const res = await fetchFn(url);
  if (!res.ok) {
    throw new Error(`GET ${url} returned HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Creates a token counter that downloads each model's tokenizer once from
 * Voyage's Hugging Face repository (the same files the official SDK uses) and
 * counts with it locally.
 */
export function createTokenCounter(
  fetchFn: typeof fetch = fetch
): TokenCounter {
  const tokenizers = new Map<string, Promise<Tokenizer>>();
  const load = (model: string): Promise<Tokenizer> => {
    let loaded = tokenizers.get(model);
    if (!loaded) {
      loaded = (async () => {
        const { Tokenizer } = await import('@huggingface/tokenizers');
        const [json, config] = await Promise.all([
          fetchJson(
            fetchFn,
            `${TOKENIZER_URL}/${model}/resolve/main/tokenizer.json`
          ),
          fetchJson(
            fetchFn,
            `${TOKENIZER_URL}/${model}/resolve/main/tokenizer_config.json`
          ),
        ]);
        return new Tokenizer(json as object, config as object);
      })();
      // A failed download is retried on the next call.
      loaded.catch(() => tokenizers.delete(model));
      tokenizers.set(model, loaded);
    }
    return loaded;
  };
  return async (model, texts) => {
    const tokenizer = await load(model);
    return texts.map((text) => tokenizer.encode(text).ids.length);
  };
}

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
import { genkit, type Genkit } from 'genkit';
import { voyageai, type VoyageAIPluginOptions } from '../src/index.js';

export interface RecordedRequest {
  url: string;
  headers: Record<string, string>;
  body: any;
}

export type Responder = (
  req: RecordedRequest
) =>
  | { status?: number; body: unknown }
  | Promise<{ status?: number; body: unknown }>;

/** A Genkit instance whose Voyage plugin talks to a recording fake instead of the network. */
export function setup(
  responder: Responder,
  pluginOptions: Partial<VoyageAIPluginOptions> = {}
): {
  ai: Genkit;
  requests: RecordedRequest[];
} {
  const requests: RecordedRequest[] = [];
  const fakeFetch = (async (input: any, init?: RequestInit) => {
    const headers: Record<string, string> = {};
    new Headers(init?.headers).forEach((v, k) => (headers[k] = v));
    const req: RecordedRequest = {
      url: typeof input === 'string' ? input : String(input?.url ?? input),
      headers,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    };
    requests.push(req);
    const res = await responder(req);
    return new Response(JSON.stringify(res.body), {
      status: res.status ?? 200,
      headers: { 'content-type': 'application/json' },
    });
  }) as typeof fetch;
  const ai = genkit({
    plugins: [
      voyageai({
        apiKey: 'test-key',
        maxRetries: 0,
        fetch: fakeFetch,
        ...pluginOptions,
      }),
    ],
  });
  return { ai, requests };
}

/** An embeddings response with `count` vectors of `dim` numbers. */
export function embeddingsBody(count: number, dim = 4) {
  return {
    object: 'list',
    data: Array.from({ length: count }, (_, index) => ({
      object: 'embedding',
      embedding: Array.from({ length: dim }, (_, i) => (index + 1) / (i + 2)),
      index,
    })),
    usage: { total_tokens: 5 },
  };
}

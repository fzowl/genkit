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
// Test app for the Voyage AI plugin. Needs the VOYAGE_API_KEY environment variable.
// Every flow takes the model explicitly: the plugin has no default model.
//
//   cd js/testapps/voyageai && genkit start -- tsx --watch src/index.ts

import { voyageai } from '@genkit-ai/voyageai';
import { Document, genkit, z } from 'genkit';
import type { Embedding } from 'genkit/embedder';

const ai = genkit({ plugins: [voyageai()] });

const InputType = z.enum(['document', 'query']);

export const embedText = ai.defineFlow(
  {
    name: 'embedText',
    inputSchema: z.object({
      model: z.string().describe('For example voyage-4, voyage-code-3'),
      texts: z.array(z.string()).min(1),
      inputType: InputType.optional(),
      outputDimension: z.number().optional(),
    }),
    outputSchema: z.object({
      count: z.number(),
      dimensions: z.number(),
    }),
  },
  async ({ model, texts, inputType, outputDimension }) => {
    const embeddings = await ai.embedMany({
      embedder: voyageai.embedder(model),
      content: texts,
      options: { inputType, outputDimension },
    });
    return {
      count: embeddings.length,
      dimensions: embeddings[0].embedding.length,
    };
  }
);

export const embedContextualized = ai.defineFlow(
  {
    name: 'embedContextualized',
    inputSchema: z.object({
      model: z.string().describe('voyage-context-4 or voyage-context-3'),
      texts: z.array(z.string()).min(1),
      inputType: InputType,
      outputDimension: z.number().optional(),
    }),
    outputSchema: z.object({
      chunks: z.array(
        z.object({
          documentIndex: z.number(),
          chunkIndex: z.number(),
          text: z.string().optional(),
          dimensions: z.number(),
        })
      ),
    }),
  },
  async ({ model, texts, inputType, outputDimension }) => {
    const embeddings = await ai.embedMany({
      embedder: voyageai.embedder(model),
      content: texts,
      options: { inputType, outputDimension },
    });
    return {
      // The metadata is on every returned embedding; `embedMany` types it narrower.
      chunks: (embeddings as Embedding[]).map((e) => ({
        documentIndex: e.metadata?.documentIndex as number,
        chunkIndex: e.metadata?.chunkIndex as number,
        text: e.metadata?.text as string | undefined,
        dimensions: e.embedding.length,
      })),
    };
  }
);

export const embedMultimodal = ai.defineFlow(
  {
    name: 'embedMultimodal',
    inputSchema: z.object({
      model: z
        .string()
        .describe('voyage-multimodal-3.5 or voyage-multimodal-3'),
      inputs: z
        .array(
          z.object({
            text: z.string().optional(),
            imageUrl: z.string().optional(),
            imageDataUrl: z
              .string()
              .optional()
              .describe('data:image/png;base64,...'),
          })
        )
        .min(1),
      inputType: InputType.optional(),
    }),
    outputSchema: z.object({ count: z.number(), dimensions: z.number() }),
  },
  async ({ model, inputs, inputType }) => {
    const documents = inputs.map((input) => {
      const content: Document['content'] = [];
      if (input.text) content.push({ text: input.text });
      if (input.imageUrl) {
        content.push({
          media: { url: input.imageUrl, contentType: 'image/jpeg' },
        });
      }
      if (input.imageDataUrl) {
        content.push({ media: { url: input.imageDataUrl } });
      }
      return Document.fromParts(content);
    });
    const embeddings = await ai.embedMany({
      embedder: voyageai.embedder(model),
      content: documents,
      options: { inputType },
    });
    return {
      count: embeddings.length,
      dimensions: embeddings[0].embedding.length,
    };
  }
);

export const rerank = ai.defineFlow(
  {
    name: 'rerank',
    inputSchema: z.object({
      model: z.string().describe('For example rerank-3'),
      query: z.string(),
      documents: z.array(z.string()).min(1),
      topK: z.number().optional(),
    }),
    outputSchema: z.array(z.object({ text: z.string(), score: z.number() })),
  },
  async ({ model, query, documents, topK }) => {
    const ranked = await ai.rerank({
      reranker: voyageai.reranker(model),
      query,
      documents: documents.map((text) => Document.fromText(text)),
      options: { topK },
    });
    return ranked.map((d) => ({ text: d.text, score: d.score() }));
  }
);

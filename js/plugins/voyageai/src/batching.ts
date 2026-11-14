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
import { GenkitError } from 'genkit';
import { MAX_INPUTS_PER_REQUEST } from './catalog.js';
import { invalidArgument } from './shared.js';

/** Counts the tokens of each text with the tokenizer of a Voyage model. */
export type TokenCounter = (
  model: string,
  texts: string[]
) => Promise<number[]>;

/** Limits Voyage enforces on one request. */
export interface BatchLimits {
  /** Maximum number of inputs. */
  maxInputs: number;
  /** Maximum total tokens; undefined when Voyage documents no ceiling. */
  maxTokens?: number;
}

/** A half-open range `[start, end)` of inputs sent in one request. */
export interface Batch {
  start: number;
  end: number;
}

/**
 * Splits inputs, in order, into consecutive batches that each respect the
 * input and token limits. `costs[i]` is the token cost of input `i`. An input
 * that cannot fit any batch is an error: it is never truncated silently.
 */
export function packBatches(
  costs: number[],
  limits: BatchLimits,
  actionName: string
): Batch[] {
  const batches: Batch[] = [];
  let start = 0;
  let tokens = 0;
  for (let i = 0; i < costs.length; i++) {
    const cost = costs[i];
    if (limits.maxTokens !== undefined && cost > limits.maxTokens) {
      throw invalidArgument(
        `${actionName}: input ${i} has ${cost} tokens, more than the ${limits.maxTokens} tokens Voyage accepts in one request.`
      );
    }
    const full =
      i - start >= limits.maxInputs ||
      (limits.maxTokens !== undefined && tokens + cost > limits.maxTokens);
    if (full) {
      batches.push({ start, end: i });
      start = i;
      tokens = 0;
    }
    tokens += cost;
  }
  if (costs.length > start) batches.push({ start, end: costs.length });
  return batches;
}

/** What is known about one input when its token cost is estimated. */
export interface CostInput {
  /** The text segments of the input (each counted separately); may be empty. */
  texts: string[];
  /** Whether the input carries images or video, whose tokens cannot be known client side. */
  hasMedia?: boolean;
}

export interface CostOptions {
  model: string;
  /** Context length of one input, in tokens. */
  contextTokens: number;
  /** Whether Voyage truncates inputs to the context length (a longer input then costs the context length). */
  truncates: boolean;
  /** Total tokens per request, if documented. */
  maxTokens?: number;
  countTokens: TokenCounter;
}

/**
 * Token cost of each input, for packing requests.
 *
 * - No documented ceiling: nothing to pack by, so nothing is counted.
 * - Inputs with media cost the context length: image and video tokens depend
 *   on pixels that are not known here, and the context length is the most one
 *   input can use.
 * - Voyage's tokenizers normalize text to NFC and then split it into byte-level
 *   pieces, so a text has at most one token per UTF-8 byte of its NFC form (plus
 *   two as slack). When those bounds already fit one request the tokenizer is
 *   not needed.
 * - Otherwise the tokens are counted with the model's tokenizer: its exact
 *   count, which in our checks was never below the server's.
 */
export async function inputCosts(
  inputs: CostInput[],
  options: CostOptions
): Promise<number[] | undefined> {
  if (options.maxTokens === undefined) return undefined;
  const cap = options.truncates ? options.contextTokens : Infinity;
  const bounds = inputs.map((input) =>
    input.hasMedia
      ? options.contextTokens
      : Math.min(
          input.texts.reduce(
            (sum, text) =>
              sum + Buffer.byteLength(text.normalize('NFC'), 'utf8') + 2,
            0
          ),
          cap
        )
  );
  if (bounds.reduce((a, b) => a + b, 0) <= options.maxTokens) return bounds;

  const toCount = inputs
    .map((input, index) => ({ input, index }))
    .filter(({ input }) => !input.hasMedia);
  // Inputs with media have a fixed cost: nothing to count.
  if (toCount.length === 0) return bounds;
  const segments = toCount.flatMap(({ input }) => input.texts);
  let counted: number[];
  try {
    counted = await options.countTokens(options.model, segments);
  } catch (e) {
    throw new GenkitError({
      status: 'UNAVAILABLE',
      message: `Could not count tokens with the ${options.model} tokenizer to split the input into token-limited requests: ${e instanceof Error ? e.message : String(e)}`,
      cause: e,
    });
  }
  if (counted.length !== segments.length) {
    throw new Error(
      `Token counter returned ${counted.length} counts for ${segments.length} texts.`
    );
  }
  const costs = [...bounds];
  let next = 0;
  for (const { input, index } of toCount) {
    let tokens = 0;
    for (let i = 0; i < input.texts.length; i++) tokens += counted[next++];
    costs[index] = Math.min(tokens, cap);
  }
  return costs;
}

/** Splits `count` inputs into batches by the cost of each input (or by input count alone when no costs). */
export function planBatches(
  count: number,
  costs: number[] | undefined,
  maxTokens: number | undefined,
  actionName: string
): Batch[] {
  return packBatches(
    costs ?? new Array<number>(count).fill(0),
    { maxInputs: MAX_INPUTS_PER_REQUEST, maxTokens },
    actionName
  );
}

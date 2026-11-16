// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/* ------------------------------------------------------------------ */
/* Added for free-model-aggregator-01: the other upstream wires.       */
/*                                                                     */
/* OpenAI client -> Anthropic provider, Anthropic client -> Anthropic  */
/* provider (pass-through), and both client wires -> Gemini provider,  */
/* with the answers translated back. Everything stays a pure function  */
/* of its arguments: no clock, no randomness.                          */
/* ------------------------------------------------------------------ */

/** A fixed epoch second for translated OpenAI completions. */
const FIXED_TRANSLATED_CREATED = 1767225600;

/** The finish_reason an Anthropic stop_reason names on the OpenAI wire. */
export function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** The stop_reason a Gemini finishReason names on the Anthropic wire. */
export function geminiStopToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The finish_reason a Gemini finishReason names on the OpenAI wire. */
export function geminiStopToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** One OpenAI content part as an Anthropic content block (text or image); null for the rest. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (dataUrl !== null) {
      return { type: 'image', source: { type: 'base64', media_type: dataUrl[1] ?? 'image/png', data: dataUrl[2] ?? '' } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** The stop sequences of an OpenAI request body, as a string array. */
function openAIStops(body: Json): string[] {
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.flatMap((s) => (typeof s === 'string' ? [s] : []));
  return [];
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. The system
 * prompt becomes `system`, image_url parts become image blocks, tool messages become tool_result
 * blocks folded into the user message they follow, and tools / tool_choice / max_tokens / stop
 * sequences pass through. response_format cannot travel on this wire and is dropped: never route a
 * JSON-mode OpenAI request to an Anthropic provider.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const pushUser = (content: Json[]): void => {
    if (content.length === 0) return;
    const last = messages[messages.length - 1];
    if (last !== undefined && last.role === 'user' && Array.isArray(last.content)) {
      (last.content as Json[]).push(...content);
      return;
    }
    messages.push({ role: 'user', content });
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      pushUser([{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: text }]);
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (typeof content === 'string') {
      pushUser([{ type: 'text', text: content }]);
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = anthropicBlockOf(part);
      if (block !== null) blocks.push(block);
    }
    pushUser(blocks);
  }
  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages answer as an OpenAI Chat Completions answer naming `model`. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content: Json[] = [];
  const blocks = Array.isArray(body.content) ? body.content : [];
  const textParts: string[] = [];
  const calls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
    if (block.type === 'tool_use') {
      calls.push({
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: `chatcmpl-${typeof body.id === 'string' ? body.id.replace(/^msg_/, '') : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_TRANSLATED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: textParts.join(''), ...(calls.length > 0 ? { tool_calls: calls } : {}) },
        finish_reason: anthropicStopToOpenAI(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** Split a raw SSE byte stream into frames: {event, data} pairs, in order. */
async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<{ event: string | null; data: string }> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        let event: string | null = null;
        const dataLines: string[] = [];
        for (const line of frame.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
        }
        yield { event, data: dataLines.join('\n') };
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      let event: string | null = null;
      const dataLines: string[] = [];
      for (const line of buffer.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
      }
      yield { event, data: dataLines.join('\n') };
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * An Anthropic SSE stream as OpenAI Chat Completions chunks: the role chunk, content deltas,
 * tool-call chunks, then one finish chunk, one usage chunk and [DONE]. `model` is named back.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  const toolIndexes = new Map<string, number>();
  let nextIndex = 0;
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: FIXED_TRANSLATED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
      if (!started) {
        started = true;
        yield chunk({ role: 'assistant', content: '' }, null);
      }
      continue;
    }
    if (frame.event === 'content_block_start') {
      const block = isRecord(data.content_block) ? data.content_block : {};
      if (block.type === 'tool_use') {
        const id = typeof block.id === 'string' ? block.id : `call_${nextIndex}`;
        toolIndexes.set(id, nextIndex);
        if (!started) {
          started = true;
          yield chunk({ role: 'assistant', content: '' }, null);
        }
        yield chunk({ tool_calls: [{ index: nextIndex, id, type: 'function', function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' } }] }, null);
        nextIndex += 1;
      }
      continue;
    }
    if (frame.event === 'content_block_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (!started) {
        started = true;
        yield chunk({ role: 'assistant', content: '' }, null);
      }
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        yield chunk({ content: delta.text }, null);
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
        const index = typeof data.index === 'number' ? data.index : 0;
        yield chunk({ tool_calls: [{ index, function: { arguments: delta.partial_json } }] }, null);
      }
      continue;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') pendingStop = anthropicStopToOpenAI(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
      continue;
    }
    if (frame.event === 'message_stop') {
      if (finished) continue;
      finished = true;
      if (!started) {
        started = true;
        yield chunk({ role: 'assistant', content: '' }, null);
      }
      yield chunk({}, pendingStop ?? 'stop');
      yield sseEvent(null, { id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: FIXED_TRANSLATED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } });
      yield 'data: [DONE]\n\n';
    }
  }
  if (!finished) {
    if (!started) yield chunk({ role: 'assistant', content: '' }, null);
    yield chunk({}, pendingStop ?? 'stop');
    yield sseEvent(null, { id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: FIXED_TRANSLATED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } });
    yield 'data: [DONE]\n\n';
  }
}

/* ------------------------------- Gemini -------------------------------- */

/** One Gemini content part's text, or '' when it holds none. */
function geminiTextOf(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
}

/** An OpenAI message's content as Gemini parts (text and inline images). */
function geminiPartsOfOpenAI(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string' && part.text !== '') parts.push({ text: part.text });
    if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
      if (dataUrl !== null) parts.push({ inlineData: { mimeType: dataUrl[1] ?? 'image/png', data: dataUrl[2] ?? '' } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

/** An Anthropic message's content as Gemini parts (text, inline images, tool calls). */
function geminiPartsOfAnthropic(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') parts.push({ text: block.text });
    if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        parts.push({ fileData: { fileUri: block.source.url } });
      }
    }
    if (block.type === 'tool_use') {
      parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
    }
  }
  return parts;
}

/** OpenAI tools as Gemini functionDeclarations. */
function geminiDeclarationsOfOpenAI(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
}

/** Anthropic tools as Gemini functionDeclarations. */
function geminiDeclarationsOfAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
}

interface GeminiRequestParts {
  systemText: string;
  contents: Json[];
  declarations: Json[];
  maxTokens: number | undefined;
  temperature: number | undefined;
  topP: number | undefined;
  stops: string[];
}

/** The shared tail of a Gemini request: tools and generationConfig. */
function geminiRequestTail(parts: GeminiRequestParts, jsonMode: boolean, model: string): Json {
  const out: Json = { model };
  if (parts.systemText !== '') out.systemInstruction = { parts: [{ text: parts.systemText }] };
  out.contents = parts.contents;
  if (parts.declarations.length > 0) out.tools = [{ functionDeclarations: parts.declarations }];
  const config: Json = {};
  if (parts.maxTokens !== undefined) config.maxOutputTokens = parts.maxTokens;
  if (parts.temperature !== undefined) config.temperature = parts.temperature;
  if (parts.topP !== undefined) config.topP = parts.topP;
  if (parts.stops.length > 0) config.stopSequences = parts.stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An OpenAI Chat Completions request as a Gemini generateContent request naming `model`. */
export function openAIToGeminiRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      contents.push({ role: 'function', parts: [{ functionResponse: { name: '', response: { content: text } } }] });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts = geminiPartsOfOpenAI(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return geminiRequestTail(
    {
      systemText: systemParts.join('\n'),
      contents,
      declarations: geminiDeclarationsOfOpenAI(body.tools),
      maxTokens,
      temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
      topP: typeof body.top_p === 'number' ? body.top_p : undefined,
      stops: openAIStops(body),
    },
    format === 'json_object' || format === 'json_schema',
    model,
  );
}

/** An Anthropic Messages request as a Gemini generateContent request naming `model`. */
export function anthropicToGeminiRequest(body: Json, model: string): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts = geminiPartsOfAnthropic(raw.content);
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    const content = raw.content;
    if (!Array.isArray(content)) {
      const parts = geminiPartsOfAnthropic(content);
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: toolResultText(block.content) } } });
        continue;
      }
      parts.push(...geminiPartsOfAnthropic([block]));
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  return geminiRequestTail(
    {
      systemText: textOf(body.system),
      contents,
      declarations: geminiDeclarationsOfAnthropic(body.tools),
      maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : undefined,
      temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
      topP: typeof body.top_p === 'number' ? body.top_p : undefined,
      stops: Array.isArray(body.stop_sequences) ? body.stop_sequences.flatMap((s) => (typeof s === 'string' ? [s] : [])) : [],
    },
    false,
    model,
  );
}

/** The first candidate's parts and finish reason of a Gemini answer, or nulls. */
function geminiCandidate(body: Json): { parts: Json[]; finish: unknown; usage: Json } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return { parts, finish: first.finishReason, usage: isRecord(body.usageMetadata) ? body.usageMetadata : {} };
}

/** A Gemini answer as an OpenAI Chat Completions answer naming `model`. */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const { parts, finish, usage } = geminiCandidate(body);
  const textParts: string[] = [];
  const calls: Json[] = [];
  parts.forEach((part, i) => {
    if (typeof part.text === 'string' && part.text !== '') textParts.push(part.text);
    if (isRecord(part.functionCall)) {
      calls.push({
        id: `call_gemini_${i}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  });
  const input = numberOrZero(usage.promptTokenCount);
  const output = numberOrZero(usage.candidatesTokenCount);
  return {
    id: `chatcmpl-gemini`,
    object: 'chat.completion',
    created: FIXED_TRANSLATED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: textParts.join(''), ...(calls.length > 0 ? { tool_calls: calls } : {}) },
        finish_reason: geminiStopToOpenAI(finish),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini answer as an Anthropic Messages answer naming `model`. */
export function geminiToAnthropicResponse(body: Json, model: string): Json {
  const { parts, finish, usage } = geminiCandidate(body);
  const content: Json[] = [];
  parts.forEach((part, i) => {
    if (typeof part.text === 'string' && part.text !== '') content.push({ type: 'text', text: part.text });
    if (isRecord(part.functionCall)) {
      content.push({
        type: 'tool_use',
        id: `toolu_gemini_${i}`,
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        input: isRecord(part.functionCall.args) ? part.functionCall.args : {},
      });
    }
  });
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: geminiStopToAnthropic(finish),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/** One Gemini SSE frame's text, tool calls, finish reason and usage, as parsed parts. */
function geminiFrameParts(data: Json): { text: string; calls: Json[]; finish: unknown; usage: Json } {
  const { parts, finish, usage } = geminiCandidate(data);
  let text = '';
  const calls: Json[] = [];
  parts.forEach((part, i) => {
    if (typeof part.text === 'string' && part.text !== '') text += part.text;
    if (isRecord(part.functionCall)) {
      calls.push({
        id: `call_gemini_${i}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  });
  return { text, calls, finish, usage };
}

/** A Gemini SSE stream as OpenAI Chat Completions chunks, ending with [DONE]. */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  let toolIndex = 0;
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: FIXED_TRANSLATED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const { text, calls, finish, usage } = geminiFrameParts(data);
    if (!started) {
      started = true;
      yield chunk({ role: 'assistant', content: '' }, null);
    }
    if (text !== '') yield chunk({ content: text }, null);
    for (const call of calls) {
      yield chunk({ tool_calls: [{ index: toolIndex, ...(isRecord(call) ? call : {}) }] }, null);
      toolIndex += 1;
    }
    if (finish !== undefined && finish !== null) pendingStop = geminiStopToOpenAI(finish);
    if (isRecord(usage) && typeof usage.promptTokenCount === 'number') {
      inputTokens = numberOrZero(usage.promptTokenCount);
      outputTokens = numberOrZero(usage.candidatesTokenCount);
    }
    if (pendingStop !== null) {
      finished = true;
      yield chunk({}, pendingStop);
      yield sseEvent(null, { id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: FIXED_TRANSLATED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } });
      yield 'data: [DONE]\n\n';
    }
  }
  if (!finished) {
    if (!started) yield chunk({ role: 'assistant', content: '' }, null);
    yield chunk({}, pendingStop ?? 'stop');
    yield sseEvent(null, { id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: FIXED_TRANSLATED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } });
    yield 'data: [DONE]\n\n';
  }
}

/** A Gemini SSE stream as Anthropic message events, ending with message_stop. */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let textIndex = -1;
  let nextIndex = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const { text, calls, finish, usage } = geminiFrameParts(data);
    if (!started) {
      started = true;
      yield sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      });
    }
    if (text !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text } });
    }
    for (const call of calls) {
      const index = nextIndex;
      nextIndex += 1;
      if (textIndex !== -1) {
        yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
        textIndex = -1;
      }
      const fn = isRecord(call) && isRecord(call.function) ? call.function : {};
      yield sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: typeof call.id === 'string' ? call.id : `toolu_${index}`, name: typeof fn.name === 'string' ? fn.name : '', input: {} },
      });
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } });
      }
      yield sseEvent('content_block_stop', { type: 'content_block_stop', index });
    }
    if (finish !== undefined && finish !== null) pendingStop = geminiStopToAnthropic(finish);
    if (isRecord(usage) && typeof usage.candidatesTokenCount === 'number') outputTokens = numberOrZero(usage.candidatesTokenCount);
    if (pendingStop !== null) {
      finished = true;
      if (textIndex !== -1) {
        yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
        textIndex = -1;
      }
      yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: pendingStop, stop_sequence: null }, usage: { output_tokens: outputTokens } });
      yield sseEvent('message_stop', { type: 'message_stop' });
    }
  }
  if (!finished) {
    if (!started) {
      yield sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      });
    }
    if (textIndex !== -1) yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
    yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: pendingStop ?? 'end_turn', stop_sequence: null }, usage: { output_tokens: outputTokens } });
    yield sseEvent('message_stop', { type: 'message_stop' });
  }
}

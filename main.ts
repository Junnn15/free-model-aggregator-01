// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// Built on the starter kit: the brief, the catalog probe and the OpenAI<->Anthropic translation
// are the starter's. What this entry adds:
//   - quality routing: every free model meeting a request's requirements is ranked by the quality
//     table's rating_milli, best first, with per-provider call counts breaking ties so the load
//     spreads across providers instead of hammering the best one;
//   - failover: a provider call refused with 402, 429 or a 5xx (or unreachable) is retried on the
//     next-ranked candidate; a provider that fails twice in a row is set aside for the trial;
//   - the other upstream wires: Anthropic providers (zai) and Gemini providers, both directions,
//     with the answers translated back into the wire the client used, streamed or not as asked;
//   - JSON-mode OpenAI requests never go to an Anthropic wire, which cannot echo response_format;
//     Gemini carries it as responseMimeType, OpenAI as response_format.
// Everything is a pure function of the brief, the requests and the providers' answers: no clock,
// no unseeded randomness, so a replay reproduces the same state export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** Per-provider routing state, derived only from the answers seen so far. */
interface ProviderState {
  /** Inference attempts made (successes and refusals alike). */
  calls: number;
  /** Refusals in a row; a success resets it. Two in a row sets the provider aside. */
  consecutiveFailures: number;
  /** Set aside for the rest of the trial after two refusals in a row. */
  dead: boolean;
}
const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, consecutiveFailures: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  // One provider's probe must never take down the whole catalog: a probe that throws
  // (network error, malformed response) is caught per provider and reported as failed.
  probes = await Promise.all(brief.directory.providers.map(async (provider) => {
    try {
      return await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
    } catch (error) {
      return { ok: false, provider, models: [], error: `probe threw: ${error instanceof Error ? error.message : String(error)}` } as Probe;
    }
  }));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, deduplicated by id, in the brief's order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every (provider, model) that can legally serve a request, best first: the quality table's rating,
 * then the provider with fewer calls so far (the load spreads), then provider and model id for a
 * deterministic order. A JSON-mode OpenAI request never goes to an Anthropic wire: it cannot carry
 * response_format, so the provider would echo json_mode false and the request would fail its checks.
 * Providers set aside after two refusals in a row are skipped.
 */
function candidatesFor(requires: Requirement, clientWire: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (stateOf(probe.provider.id).dead) continue;
    if (clientWire === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const callsA = stateOf(a.probe.provider.id).calls;
    const callsB = stateOf(b.probe.provider.id).calls;
    if (callsA !== callsB) return callsA - callsB;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether a client request on the OpenAI wire asks for JSON mode. */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The upstream request for one candidate: the provider's wire, the client's body translated. */
function upstreamCall(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const model = candidate.model.id;
  const stream = body.stream === true;
  if (provider.wire === 'openai') {
    const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
    if (stream) headers.accept = 'text/event-stream';
    let upstreamBody: Json;
    if (clientWire === 'openai') {
      upstreamBody = { ...body, model };
      if (stream) {
        // The client checks the stream's usage: ask the provider to include it.
        const options = isRecord(body.stream_options) ? { ...body.stream_options } : {};
        options.include_usage = true;
        upstreamBody.stream_options = options;
      }
    } else {
      upstreamBody = anthropicToOpenAI(body, model);
    }
    return { url: `${base}/v1/chat/completions`, headers, body: upstreamBody };
  }
  if (provider.wire === 'anthropic') {
    const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key), 'anthropic-version': '2023-06-01' };
    if (stream) headers.accept = 'text/event-stream';
    const upstreamBody = clientWire === 'openai' ? openAIToAnthropicRequest(body, model) : { ...body, model };
    return { url: `${base}/v1/messages`, headers, body: upstreamBody };
  }
  // Gemini wire.
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  const upstreamBody = clientWire === 'openai' ? openAIToGeminiRequest(body, model) : anthropicToGeminiRequest(body, model);
  const method = stream ? 'streamGenerateContent' : 'generateContent';
  return { url: `${base}/v1beta/models/${model}:${method}`, headers, body: upstreamBody };
}

/** Record one provider attempt: successes reset the failure streak, refusals grow it. */
function recordAttempt(providerId: string, ok: boolean): void {
  const state = stateOf(providerId);
  state.calls += 1;
  if (ok) {
    state.consecutiveFailures = 0;
    return;
  }
  state.consecutiveFailures += 1;
  if (state.consecutiveFailures >= 2) state.dead = true;
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/**
 * Serve one client request against the ranked candidates, in order, until one answers. A refusal
 * (402, 429, 5xx, unreachable) moves to the next candidate; only the first success is returned, so
 * exactly one provider call answers each request. Returns true when an answer was sent.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse, candidates: Candidate[]): Promise<boolean> {
  const stream = body.stream === true;
  for (const candidate of candidates) {
    const call = upstreamCall(candidate, clientWire, body);
    let upstream: Response;
    try {
      upstream = await fetch(call.url, {
        method: 'POST',
        headers: call.headers,
        body: JSON.stringify(call.body),
        signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
      });
    } catch (err) {
      log(`${candidate.probe.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      recordAttempt(candidate.probe.provider.id, false);
      continue;
    }
    if (!upstream.ok) {
      // Drain the error body, then fail over: the refusal is not the answer.
      try {
        await upstream.text();
      } catch {
        // The body is unreadable: the refusal stands either way.
      }
      log(`${candidate.probe.provider.id}/${candidate.model.id} refused HTTP ${upstream.status}`);
      recordAttempt(candidate.probe.provider.id, false);
      continue;
    }
    recordAttempt(candidate.probe.provider.id, true);
    const wire = candidate.probe.provider.wire;
    if (stream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (clientWire === 'openai') {
        if (wire === 'openai') await pipeStream(upstream.body, res);
        else if (wire === 'anthropic') {
          for await (const event of openAIEventsFromAnthropic(upstream.body, candidate.model.id)) res.write(event);
          res.end();
        } else {
          for await (const event of openAIEventsFromGemini(upstream.body, candidate.model.id)) res.write(event);
          res.end();
        }
      } else if (wire === 'openai') {
        const requested = typeof body.model === 'string' && body.model !== '' ? body.model : candidate.model.id;
        for await (const event of anthropicEventsFromOpenAI(upstream.body, requested)) res.write(event);
        res.end();
      } else if (wire === 'anthropic') {
        await pipeStream(upstream.body, res);
      } else {
        for await (const event of anthropicEventsFromGemini(upstream.body, candidate.model.id)) res.write(event);
        res.end();
      }
      return true;
    }
    const answer: unknown = await upstream.json();
    if (!isRecord(answer)) {
      // A 2xx with no JSON body is not a usable answer: treat it as a failure and move on.
      log(`${candidate.probe.provider.id}/${candidate.model.id} answered 2xx with no JSON body`);
      recordAttempt(candidate.probe.provider.id, false);
      continue;
    }
    if (clientWire === 'openai') {
      if (wire === 'openai') sendJson(res, 200, answer);
      else if (wire === 'anthropic') sendJson(res, 200, anthropicToOpenAIResponse(answer, candidate.model.id));
      else sendJson(res, 200, geminiToOpenAIResponse(answer, candidate.model.id));
    } else if (wire === 'openai') {
      const requested = typeof body.model === 'string' && body.model !== '' ? body.model : candidate.model.id;
      sendJson(res, 200, openAIToAnthropic(answer, requested));
    } else if (wire === 'anthropic') {
      sendJson(res, 200, answer);
    } else {
      sendJson(res, 200, geminiToAnthropicResponse(answer, candidate.model.id));
    }
    return true;
  }
  return false;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that fits. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body), 'openai', jsonModeOf(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  if (await serve('openai', body, res, candidates)) return;
  sendError(res, 'openai', 502, 'upstream_unreachable', 'every provider that could serve this request refused it');
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that fits. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body), 'anthropic', false);
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  if (await serve('anthropic', body, res, candidates)) return;
  sendError(res, 'anthropic', 502, 'api_error', 'every provider that could serve this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

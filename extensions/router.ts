import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEvent,
	type AssistantMessageEventStream,
	type Context,
	createAssistantMessageEventStream,
	type Message,
	type Model,
	openAICompletionsApi,
	type SimpleStreamOptions,
	type Usage,
} from "@earendil-works/pi-ai/compat";

const PROVIDER_ID = "local-router";
const ROUTER_MODEL_ID = "local-to-hf";
const ROUTER_PROVIDER_API_KEY = "pi-local-router";
const LOCAL_MODEL = process.env.ROUTER_LOCAL_MODEL ?? "local-model";
const LOCAL_BASE_URL = trimTrailingSlash(process.env.ROUTER_LOCAL_BASE_URL ?? "http://127.0.0.1:8080/v1");
const REMOTE_MODEL = process.env.ROUTER_REMOTE_MODEL ?? "zai-org/GLM-5.2:fastest";
const REMOTE_BASE_URL = trimTrailingSlash(process.env.ROUTER_REMOTE_BASE_URL ?? "https://router.huggingface.co/v1");
const LOCAL_TIMEOUT_MS = envInt("ROUTER_DECISION_TIMEOUT_MS", 120_000);
const LOCAL_MAX_TOKENS = envInt("ROUTER_LOCAL_MAX_TOKENS", 512);
const ROUTER_CONTEXT_WINDOW = envInt("ROUTER_CONTEXT_WINDOW", 32_768);
const ROUTER_MAX_TOKENS = envInt("ROUTER_MAX_TOKENS", 8_192);
const ENTROPY_THRESHOLD = envFloat("ROUTER_ENTROPY_THRESHOLD", 0.12);
const TOP1_THRESHOLD = envFloat("ROUTER_TOP1_THRESHOLD", 0.95);
const CONFIDENCE_THRESHOLD = envFloat("ROUTER_CONFIDENCE_THRESHOLD", 0.97);
const FAIL_OPEN_TO_REMOTE = (process.env.ROUTER_FAIL_OPEN ?? "remote").toLowerCase() === "remote";
const SHOW_TRACE = envBool("ROUTER_SHOW_TRACE", true);
const ROUTER_TRACE_LINE_RE = /^\s*[>|│]?\s*router:\s*route=.*(?:\n|$)/gim;

type Route = "local" | "remote";

interface RouterDecision {
	route: Route;
	text?: string;
	model?: string;
	confidence: number;
	reason?: string;
	metrics?: {
		avg_entropy?: number;
		p90_entropy?: number;
		max_token_entropy?: number;
		normalized_entropy?: number;
		mean_top1_prob?: number;
		p10_top1_prob?: number;
		min_top1_prob?: number;
		generated_tokens?: number;
	};
	usage?: {
		prompt_tokens?: number;
		completion_tokens?: number;
		total_tokens?: number;
	};
	route_source?: string;
	cache_hit?: boolean;
	latency_ms?: number;
}

function envInt(name: string, fallback: number): number {
	const raw = process.env[name];
	if (!raw) return fallback;
	const parsed = Number.parseInt(raw, 10);
	return Number.isFinite(parsed) ? parsed : fallback;
}

function envFloat(name: string, fallback: number): number {
	const raw = process.env[name];
	if (!raw) return fallback;
	const parsed = Number.parseFloat(raw);
	return Number.isFinite(parsed) ? parsed : fallback;
}

function envBool(name: string, fallback: boolean): boolean {
	const raw = process.env[name];
	if (!raw) return fallback;
	return !["0", "false", "no", "off"].includes(raw.toLowerCase());
}

function trimTrailingSlash(value: string): string {
	return value.replace(/\/+$/, "");
}

function contentToText(content: Message["content"]): string {
	if (typeof content === "string") return content;
	return content
		.map((part) => {
			if (part.type === "text") return part.text;
			if (part.type === "image") return "[image omitted for local entropy probe]";
			return "";
		})
		.filter(Boolean)
		.join("\n");
}

function stripRouterTrace(text: string): string {
	return text.replace(ROUTER_TRACE_LINE_RE, "").trim();
}

function assistantText(message: Extract<Message, { role: "assistant" }>): string {
	const text = message.content
		.map((part) => {
			if (part.type === "text") return part.text;
			if (part.type === "toolCall") return `[tool call: ${part.name}]`;
			return "";
		})
		.filter(Boolean)
		.join("\n");
	return stripRouterTrace(text);
}

function toOpenAIProbeMessages(context: Context): Array<{ role: "system" | "user" | "assistant"; content: string }> {
	const messages: Array<{ role: "system" | "user" | "assistant"; content: string }> = [];
	for (const message of context.messages) {
		if (message.role === "user") {
			messages.push({ role: "user", content: contentToText(message.content) });
		} else if (message.role === "assistant") {
			const text = assistantText(message);
			if (text) messages.push({ role: "assistant", content: text });
		} else if (message.role === "toolResult") {
			const content = contentToText(message.content);
			messages.push({
				role: "user",
				content: `[tool result: ${message.toolName}${message.isError ? ", error" : ""}]\n${content}`,
			});
		}
	}
	return messages;
}

function signalWithTimeout(parent: AbortSignal | undefined, timeoutMs: number): { signal: AbortSignal; cleanup: () => void } {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(new Error("local router decision timed out")), timeoutMs);
	const onAbort = () => controller.abort(parent?.reason);
	parent?.addEventListener("abort", onAbort, { once: true });
	return {
		signal: controller.signal,
		cleanup: () => {
			clearTimeout(timer);
			parent?.removeEventListener("abort", onAbort);
		},
	};
}

async function requestLocalDecision(context: Context, options?: SimpleStreamOptions): Promise<RouterDecision> {
	const { signal, cleanup } = signalWithTimeout(options?.signal, LOCAL_TIMEOUT_MS);
	try {
		const response = await fetch(`${LOCAL_BASE_URL}/router/decision`, {
			method: "POST",
			headers: { "content-type": "application/json" },
			signal,
			body: JSON.stringify({
				model: LOCAL_MODEL,
				messages: toOpenAIProbeMessages(context),
				max_tokens: Math.min(options?.maxTokens ?? LOCAL_MAX_TOKENS, LOCAL_MAX_TOKENS),
				temperature: options?.temperature ?? 0.2,
				entropy_threshold: ENTROPY_THRESHOLD,
				top1_threshold: TOP1_THRESHOLD,
				confidence_threshold: CONFIDENCE_THRESHOLD,
			}),
		});
		if (!response.ok) {
			throw new Error(`local router returned ${response.status}: ${await response.text()}`);
		}
		return (await response.json()) as RouterDecision;
	} finally {
		cleanup();
	}
}

function emptyUsage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function usageFromDecision(decision: RouterDecision): Usage {
	const usage = emptyUsage();
	usage.input = decision.usage?.prompt_tokens ?? 0;
	usage.output = decision.usage?.completion_tokens ?? 0;
	usage.totalTokens = decision.usage?.total_tokens ?? usage.input + usage.output;
	return usage;
}

function chunkText(text: string): string[] {
	const chunks = text.match(/\S+\s*/g);
	return chunks && chunks.length > 0 ? chunks : [text];
}

function shortText(value: string | undefined, maxLength = 140): string | undefined {
	const normalized = value?.replace(/\s+/g, " ").trim();
	if (!normalized) return undefined;
	return normalized.length > maxLength ? `${normalized.slice(0, maxLength - 3)}...` : normalized;
}

function finiteMetric(value: number | undefined): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function routeResponseModel(decision: RouterDecision): string {
	return decision.route === "local" ? decision.model ?? LOCAL_MODEL : REMOTE_MODEL;
}

function routeTraceText(decision: RouterDecision): string {
	if (!SHOW_TRACE) return "";
	const fields = [`route=${decision.route}`, `model=${routeResponseModel(decision)}`];
	const confidence = finiteMetric(decision.confidence);
	if (confidence !== undefined) fields.push(`confidence=${confidence.toFixed(3)}`);
	if (decision.route_source) fields.push(`source=${decision.route_source}`);
	if (decision.cache_hit) fields.push("cache=hit");
	const entropy = finiteMetric(decision.metrics?.avg_entropy);
	if (entropy !== undefined) fields.push(`entropy=${entropy.toFixed(3)}`);
	const p90Entropy = finiteMetric(decision.metrics?.p90_entropy);
	if (p90Entropy !== undefined) fields.push(`p90=${p90Entropy.toFixed(3)}`);
	const top1 = finiteMetric(decision.metrics?.mean_top1_prob);
	if (top1 !== undefined) fields.push(`top1=${top1.toFixed(3)}`);
	const p10Top1 = finiteMetric(decision.metrics?.p10_top1_prob);
	if (p10Top1 !== undefined) fields.push(`p10=${p10Top1.toFixed(3)}`);
	const reason = shortText(decision.reason);
	if (reason) fields.push(`reason=${reason}`);
	return `> router: ${fields.join(" | ")}\n\n`;
}

function withRouterDiagnostics(message: AssistantMessage, decision: RouterDecision): AssistantMessage {
	return {
		...message,
		responseModel: message.responseModel ?? routeResponseModel(decision),
		diagnostics: [
			...(message.diagnostics ?? []),
			{
				type: "local-router-decision",
				timestamp: Date.now(),
				details: {
					route: decision.route,
					responseModel: routeResponseModel(decision),
					decisionModel: decision.model ?? LOCAL_MODEL,
					confidence: decision.confidence,
					routeSource: decision.route_source,
					cacheHit: decision.cache_hit,
					latencyMs: decision.latency_ms,
					reason: decision.reason,
					metrics: decision.metrics,
				},
			},
		],
	};
}

function prependTrace(message: AssistantMessage, trace: string, decision: RouterDecision): AssistantMessage {
	const withDiagnostics = withRouterDiagnostics(message, decision);
	if (!trace) return withDiagnostics;
	return {
		...withDiagnostics,
		content: [{ type: "text", text: trace }, ...withDiagnostics.content],
	};
}

function prependTracePlaceholder(message: AssistantMessage, decision: RouterDecision): AssistantMessage {
	const withDiagnostics = withRouterDiagnostics(message, decision);
	return {
		...withDiagnostics,
		content: [{ type: "text", text: "" }, ...withDiagnostics.content],
	};
}

function addTraceToRemoteEvent(event: AssistantMessageEvent, trace: string, decision: RouterDecision): AssistantMessageEvent[] {
	if (!trace) {
		if (event.type === "done") return [{ ...event, message: withRouterDiagnostics(event.message, decision) }];
		if (event.type === "error") return [{ ...event, error: withRouterDiagnostics(event.error, decision) }];
		if ("partial" in event) return [{ ...event, partial: withRouterDiagnostics(event.partial, decision) }];
		return [event];
	}

	if (event.type === "start") {
		const partial = withRouterDiagnostics(event.partial, decision);
		const traceStartedPartial = prependTracePlaceholder(event.partial, decision);
		const traceEndedPartial = prependTrace(event.partial, trace, decision);
		return [
			{ type: "start", partial },
			{ type: "text_start", contentIndex: 0, partial: traceStartedPartial },
			{ type: "text_delta", contentIndex: 0, delta: trace, partial: traceEndedPartial },
			{ type: "text_end", contentIndex: 0, content: trace, partial: traceEndedPartial },
		];
	}
	if (event.type === "done") return [{ ...event, message: prependTrace(event.message, trace, decision) }];
	if (event.type === "error") return [{ ...event, error: prependTrace(event.error, trace, decision) }];
	if ("contentIndex" in event && "partial" in event) {
		return [
			{
				...event,
				contentIndex: event.contentIndex + 1,
				partial: prependTrace(event.partial, trace, decision),
			} as AssistantMessageEvent,
		];
	}
	return [event];
}

function pushError(stream: AssistantMessageEventStream, model: Model<Api>, error: unknown): void {
	stream.push({
		type: "error",
		reason: "error",
		error: {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: emptyUsage(),
			stopReason: "error",
			errorMessage: error instanceof Error ? error.message : String(error),
			timestamp: Date.now(),
		},
	});
	stream.end();
}

async function emitLocalAnswer(
	stream: AssistantMessageEventStream,
	model: Model<Api>,
	decision: RouterDecision,
): Promise<void> {
	const output: AssistantMessage = withRouterDiagnostics(
		{
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			responseModel: decision.model ?? LOCAL_MODEL,
			usage: usageFromDecision(decision),
			stopReason: "stop",
			timestamp: Date.now(),
		},
		decision,
	);

	stream.push({ type: "start", partial: output });
	output.content.push({ type: "text", text: "" });
	const contentIndex = output.content.length - 1;
	stream.push({ type: "text_start", contentIndex, partial: output });

	const block = output.content[contentIndex];
	if (block.type !== "text") throw new Error("unexpected local text block");
	for (const delta of chunkText(`${routeTraceText(decision)}${decision.text ?? ""}`)) {
		block.text += delta;
		stream.push({ type: "text_delta", contentIndex, delta, partial: output });
		await Promise.resolve();
	}

	stream.push({ type: "text_end", contentIndex, content: block.text, partial: output });
	stream.push({ type: "done", reason: "stop", message: output });
	stream.end();
}

function remoteModelFor(model: Model<Api>): Model<"openai-completions"> {
	return {
		...model,
		id: REMOTE_MODEL,
		name: `Hugging Face remote (${REMOTE_MODEL})`,
		api: "openai-completions",
		baseUrl: REMOTE_BASE_URL,
		contextWindow: ROUTER_CONTEXT_WINDOW,
		maxTokens: ROUTER_MAX_TOKENS,
	} as Model<"openai-completions">;
}

function remoteApiKey(options?: SimpleStreamOptions): string {
	const optionApiKey = options?.apiKey && options.apiKey !== ROUTER_PROVIDER_API_KEY ? options.apiKey : undefined;
	const apiKey = optionApiKey ?? process.env.HF_TOKEN ?? process.env.HUGGINGFACE_API_KEY ?? process.env.HUGGING_FACE_HUB_TOKEN;
	if (!apiKey) {
		throw new Error("remote route requires HF_TOKEN, HUGGINGFACE_API_KEY, or HUGGING_FACE_HUB_TOKEN");
	}
	return apiKey;
}

function streamEntropyRouter(model: Model<Api>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();

	(async () => {
		try {
			let decision: RouterDecision;
			try {
				decision = await requestLocalDecision(context, options);
			} catch (error) {
				if (!FAIL_OPEN_TO_REMOTE) throw error;
				decision = {
					route: "remote",
					confidence: 0,
					reason: `local decision unavailable: ${error instanceof Error ? error.message : String(error)}`,
				};
			}

			if (decision.route === "local" && decision.text) {
				await emitLocalAnswer(stream, model, decision);
				return;
			}

			const innerStream = openAICompletionsApi().streamSimple(remoteModelFor(model), context, {
				...options,
				apiKey: remoteApiKey(options),
			});
			const trace = routeTraceText(decision);
			for await (const event of innerStream) {
				for (const annotatedEvent of addTraceToRemoteEvent(event, trace, decision)) {
					stream.push(annotatedEvent);
				}
			}
			stream.end();
		} catch (error) {
			pushError(stream, model, error);
		}
	})();

	return stream;
}

export default function (pi: ExtensionAPI) {
	pi.registerProvider(PROVIDER_ID, {
		name: "Local Router",
		baseUrl: REMOTE_BASE_URL,
		apiKey: ROUTER_PROVIDER_API_KEY,
		api: "local-router-api",
		models: [
			{
				id: ROUTER_MODEL_ID,
				name: "Local model -> Hugging Face router",
				reasoning: false,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: ROUTER_CONTEXT_WINDOW,
				maxTokens: ROUTER_MAX_TOKENS,
			},
		],
		streamSimple: streamEntropyRouter,
	});

	pi.registerCommand("router-status", {
		description: "Check the local router server",
		handler: async (_args, ctx) => {
			try {
				const response = await fetch(`${LOCAL_BASE_URL.replace(/\/v1$/, "")}/health`);
				if (!response.ok) {
					ctx.ui.notify(`[local-router] local server returned ${response.status}`, "warning");
					return;
				}
				const payload = (await response.json()) as { model?: string; backend?: string; mock?: boolean };
				ctx.ui.notify(
					`[local-router] local=${payload.model ?? LOCAL_MODEL} backend=${payload.backend ?? "unknown"} mock=${String(payload.mock ?? false)} remote=${REMOTE_MODEL}`,
					"info",
				);
			} catch (error) {
				ctx.ui.notify(
					`[local-router] local server unavailable: ${error instanceof Error ? error.message : String(error)}`,
					"error",
				);
			}
		},
	});
}

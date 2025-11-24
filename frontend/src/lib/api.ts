export const API_BASE_URL = "http://localhost:8000";

export interface SingleExperimentRequest {
    task: string;
    gpu?: string;
    model?: string;
    test_mode?: boolean;
}

export interface OrchestratorExperimentRequest {
    task: string;
    gpu?: string;
    model?: string;
    num_agents: number;
    max_rounds: number;
    max_parallel: number;
    test_mode?: boolean;
}

export type ExperimentRequest =
    | SingleExperimentRequest
    | OrchestratorExperimentRequest;

export interface LogEvent {
    type: "line" | "summary";
    stream?: "stdout" | "stderr";
    timestamp: string;
    raw?: string;
    plain?: string;
    exit_code?: number;
    duration_seconds?: number;
}

export interface ChartSeries {
    name: string;
    values: number[];
}

export interface ChartSpec {
    title?: string;
    type: "line" | "bar";
    labels: string[];
    series: ChartSeries[];
}

export interface AgentSummaryRequest {
    agent_id: string;
    history: { type: "thought" | "code" | "result" | "text"; content: string }[];
}

export interface AgentSummaryResponse {
    summary: string;
    chart?: ChartSpec | null;
}

export interface CredentialStatus {
    hasGoogleApiKey: boolean;
    hasAnthropicApiKey: boolean;
    hasModalToken: boolean;
}

export interface CredentialUpdatePayload {
    googleApiKey?: string;
    anthropicApiKey?: string;
    modalTokenId?: string;
    modalTokenSecret?: string;
}

export async function streamExperiment(
    endpoint: "/api/experiments/single/stream" | "/api/experiments/orchestrator/stream",
    payload: ExperimentRequest,
    onData: (data: LogEvent) => void,
    onError: (error: Error) => void,
    onComplete: () => void
) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }

        if (!response.body) {
            throw new Error("No response body");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const event = JSON.parse(line);
                    onData(event);
                } catch (e) {
                    console.warn("Failed to parse JSON line:", line, e);
                }
            }
        }

        onComplete();
    } catch (error) {
        onError(error instanceof Error ? error : new Error(String(error)));
    }
}

export async function summarizeAgent(payload: AgentSummaryRequest): Promise<AgentSummaryResponse> {
    const response = await fetch(`${API_BASE_URL}/api/agents/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        throw new Error(`Summarizer error: ${response.status} ${response.statusText}`);
    }

    const data = (await response.json()) as AgentSummaryResponse;
    return data;
}

export async function fetchCredentialStatus(): Promise<CredentialStatus> {
    const response = await fetch(`${API_BASE_URL}/api/credentials/status`);
    if (!response.ok) {
        throw new Error(`Credential check failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    return {
        hasGoogleApiKey: Boolean(data.has_google_api_key),
        hasAnthropicApiKey: Boolean(data.has_anthropic_api_key),
        hasModalToken: Boolean(data.has_modal_token),
    };
}

export async function saveCredentials(payload: CredentialUpdatePayload): Promise<CredentialStatus> {
    const response = await fetch(`${API_BASE_URL}/api/credentials`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            google_api_key: payload.googleApiKey,
            anthropic_api_key: payload.anthropicApiKey,
            modal_token_id: payload.modalTokenId,
            modal_token_secret: payload.modalTokenSecret,
        }),
    });

    if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Unable to save credentials: ${response.status} ${response.statusText} - ${detail}`);
    }

    const data = await response.json();
    return {
        hasGoogleApiKey: Boolean(data.has_google_api_key),
        hasAnthropicApiKey: Boolean(data.has_anthropic_api_key),
        hasModalToken: Boolean(data.has_modal_token),
    };
}

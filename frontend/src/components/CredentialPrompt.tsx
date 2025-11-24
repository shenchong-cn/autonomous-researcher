import { CredentialStatus } from "@/lib/api";
import { AnimatePresence, motion } from "framer-motion";
import { ExternalLink, KeyRound, Loader2, ShieldCheck, Sparkles, X } from "lucide-react";

export type CredentialFormState = {
    googleApiKey: string;
    anthropicApiKey: string;
    modalTokenId: string;
    modalTokenSecret: string;
};

type CredentialPromptProps = {
    open: boolean;
    status: CredentialStatus | null;
    selectedModel: "gemini-3-pro-preview" | "claude-opus-4-5";
    form: CredentialFormState;
    onChange: (field: keyof CredentialFormState, value: string) => void;
    onSubmit: () => void;
    onClose: () => void;
    isSaving: boolean;
    error?: string | null;
};

export function CredentialPrompt({
    open,
    status,
    selectedModel,
    form,
    onChange,
    onSubmit,
    onClose,
    isSaving,
    error,
}: CredentialPromptProps) {
    const googleReady = !!status?.hasGoogleApiKey;
    const anthropicReady = !!status?.hasAnthropicApiKey;
    const modalReady = !!status?.hasModalToken;

    // Determine which key is needed based on selected model
    const needsGoogleKey = selectedModel === "gemini-3-pro-preview";
    const needsAnthropicKey = selectedModel === "claude-opus-4-5";

    const requiredKeyName = needsGoogleKey ? "Google API key" : "Anthropic API key";
    const hasRequiredKey = needsGoogleKey ? googleReady : anthropicReady;

    const readinessCopy =
        hasRequiredKey && modalReady
            ? "All set — keys already saved locally."
            : `Needed: ${[
                  !hasRequiredKey ? requiredKeyName : null,
                  modalReady ? null : "Modal token (id + secret)",
              ]
                  .filter(Boolean)
                  .join(" + ")}`;

    const googleProvided = !!form.googleApiKey.trim();
    const anthropicProvided = !!form.anthropicApiKey.trim();
    const modalProvided = !!form.modalTokenId.trim() && !!form.modalTokenSecret.trim();

    // Check if the required key for the selected model is available or provided
    const hasRequiredLLMKey = needsGoogleKey
        ? (googleReady || googleProvided)
        : (anthropicReady || anthropicProvided);
    const hasModalCredentials = modalReady || modalProvided;

    const disableSubmit =
        isSaving || !hasRequiredLLMKey || !hasModalCredentials;

    return (
        <AnimatePresence>
            {open && (
                <motion.div
                    className="fixed inset-0 z-[120] flex items-center justify-center bg-black/70 backdrop-blur-sm px-4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                >
                    <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_top,_rgba(255,255,255,0.08),_transparent_45%)]" />
                    <motion.div
                        initial={{ opacity: 0, y: 30, scale: 0.97 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 20 }}
                        transition={{ duration: 0.2 }}
                        className="relative w-full max-w-4xl overflow-hidden rounded-2xl border border-white/10 bg-[#0b0b0c]/90 shadow-2xl"
                    >
                        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/30 to-transparent" />
                        <div className="absolute -left-40 -top-40 h-80 w-80 rounded-full bg-purple-500/10 blur-3xl" />
                        <div className="absolute -right-32 top-10 h-64 w-64 rounded-full bg-blue-500/10 blur-3xl" />

                        <div className="relative p-8 md:p-10 space-y-6">
                            <div className="flex items-start justify-between gap-4">
                                <div className="space-y-2">
                                    <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.22em] text-white/70">
                                        <Sparkles className="h-3.5 w-3.5 text-amber-300" />
                                        Environment check
                                    </div>
                                    <h2 className="text-2xl font-semibold text-white tracking-tight">
                                        Add your API keys to launch the run
                                    </h2>
                                    <p className="text-sm text-white/60 max-w-2xl">
                                        We need at least one LLM key (Google for Gemini or Anthropic for Claude) and a Modal token pair to spin up research sandboxes. Keys are stored locally in your <code className="px-2 py-1 rounded bg-white/5 text-[11px]">.env</code>.
                                    </p>
                                </div>
                                <button
                                    onClick={onClose}
                                    className="text-white/60 hover:text-white transition-colors rounded-full p-2 border border-white/10 hover:border-white/30"
                                    aria-label="Close credential prompt"
                                >
                                    <X className="h-5 w-5" />
                                </button>
                            </div>

                            <div className="grid gap-6 md:grid-cols-[1.1fr_1.2fr]">
                                <div className="space-y-4 rounded-xl border border-white/10 bg-white/5 p-5">
                                    <div className="flex items-center gap-3">
                                        <ShieldCheck className={`h-5 w-5 ${hasRequiredKey && modalReady ? "text-emerald-400" : "text-amber-300"}`} />
                                        <div className="flex-1">
                                            <p className="text-sm font-medium text-white">Credentials status</p>
                                            <p className="text-xs text-white/60">
                                                {hasRequiredKey && modalReady
                                                    ? "Ready to launch."
                                                    : "Add the missing keys to continue."}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="flex flex-col gap-3">
                                        <StatusPill
                                            label="Google API key (Gemini)"
                                            ok={googleReady}
                                            required={needsGoogleKey}
                                        />
                                        <StatusPill
                                            label="Anthropic API key (Claude)"
                                            ok={anthropicReady}
                                            required={needsAnthropicKey}
                                        />
                                        <StatusPill label="Modal token (compute sandbox)" ok={modalReady} required={true} />
                                    </div>

                                    <div className="grid grid-cols-3 gap-3 pt-1">
                                        <a
                                            href="https://aistudio.google.com/app/apikey"
                                            target="_blank"
                                            rel="noreferrer"
                                            className="group inline-flex items-center justify-center gap-2 rounded-lg border border-white/15 bg-gradient-to-r from-white/5 to-white/0 px-3 py-3 text-xs font-medium text-white transition hover:border-white/30 hover:from-white/10"
                                        >
                                            <Sparkles className="h-4 w-4 text-amber-300" />
                                            Google key
                                            <ExternalLink className="h-3 w-3 text-white/50 group-hover:text-white" />
                                        </a>
                                        <a
                                            href="https://console.anthropic.com/settings/keys"
                                            target="_blank"
                                            rel="noreferrer"
                                            className="group inline-flex items-center justify-center gap-2 rounded-lg border border-white/15 bg-gradient-to-r from-white/5 to-white/0 px-3 py-3 text-xs font-medium text-white transition hover:border-white/30 hover:from-white/10"
                                        >
                                            <Sparkles className="h-4 w-4 text-purple-300" />
                                            Anthropic key
                                            <ExternalLink className="h-3 w-3 text-white/50 group-hover:text-white" />
                                        </a>
                                        <a
                                            href="https://modal.com/account/tokens"
                                            target="_blank"
                                            rel="noreferrer"
                                            className="group inline-flex items-center justify-center gap-2 rounded-lg border border-white/15 bg-gradient-to-r from-white/5 to-white/0 px-3 py-3 text-xs font-medium text-white transition hover:border-white/30 hover:from-white/10"
                                        >
                                            <KeyRound className="h-4 w-4 text-sky-300" />
                                            Modal tokens
                                            <ExternalLink className="h-3 w-3 text-white/50 group-hover:text-white" />
                                        </a>
                                    </div>
                                </div>

                                <div className="space-y-4 rounded-xl border border-white/10 bg-black/60 p-5">
                                    <Field
                                        label="Google API key"
                                        placeholder="Paste your AI Studio key"
                                        value={form.googleApiKey}
                                        onChange={(value) => onChange("googleApiKey", value)}
                                        status={googleReady ? "ok" : needsGoogleKey ? "missing" : "optional"}
                                        helper="Used for Gemini 3 Pro (stored locally)."
                                    />
                                    <Field
                                        label="Anthropic API key"
                                        placeholder="Paste your Anthropic key"
                                        value={form.anthropicApiKey}
                                        onChange={(value) => onChange("anthropicApiKey", value)}
                                        status={anthropicReady ? "ok" : needsAnthropicKey ? "missing" : "optional"}
                                        helper="Used for Claude Opus 4.5 (stored locally)."
                                    />
                                    <Field
                                        label="Modal token ID"
                                        placeholder="modal-token-id"
                                        value={form.modalTokenId}
                                        onChange={(value) => onChange("modalTokenId", value)}
                                        status={modalReady ? "ok" : "missing"}
                                        helper="Pair with the secret to deploy sandboxes."
                                    />
                                    <Field
                                        label="Modal token secret"
                                        placeholder="modal-token-secret"
                                        value={form.modalTokenSecret}
                                        onChange={(value) => onChange("modalTokenSecret", value)}
                                        status={modalReady ? "ok" : "missing"}
                                        helper="Kept locally in your .env file."
                                    />

                                    {error && (
                                        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
                                            {error}
                                        </div>
                                    )}

                                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between pt-1">
                                        <p className="text-xs text-white/50">{readinessCopy}</p>
                                        <button
                                            onClick={onSubmit}
                                            disabled={disableSubmit}
                                            className="inline-flex items-center justify-center gap-2 rounded-lg bg-white px-4 py-2.5 text-sm font-semibold text-black transition hover:bg-white/90 disabled:cursor-not-allowed disabled:bg-white/40"
                                        >
                                            {isSaving ? (
                                                <>
                                                    <Loader2 className="h-4 w-4 animate-spin" />
                                                    Saving...
                                                </>
                                            ) : (
                                                <>
                                                    Save &amp; continue
                                                    <ShieldCheck className="h-4 w-4" />
                                                </>
                                            )}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}

type StatusPillProps = {
    label: string;
    ok: boolean;
    required?: boolean;
};

function StatusPill({ label, ok, required = true }: StatusPillProps) {
    // If not required and not ok, show as "optional" (gray/neutral)
    const isOptional = !required && !ok;

    return (
        <div className={`flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-3 py-2 ${isOptional ? "opacity-50" : ""}`}>
            <span className="text-sm text-white/80">{label}</span>
            <span
                className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-[11px] font-semibold ${
                    ok
                        ? "bg-emerald-500/15 text-emerald-200"
                        : isOptional
                        ? "bg-white/5 text-white/40"
                        : "bg-amber-500/10 text-amber-200"
                }`}
            >
                <div
                    className={`h-2 w-2 rounded-full ${
                        ok
                            ? "bg-emerald-400 shadow-[0_0_0_4px_rgba(52,211,153,0.2)]"
                            : isOptional
                            ? "bg-white/30"
                            : "bg-amber-300 shadow-[0_0_0_4px_rgba(251,191,36,0.25)]"
                    }`}
                />
                {ok ? "Ready" : isOptional ? "Optional" : "Missing"}
            </span>
        </div>
    );
}

type FieldProps = {
    label: string;
    placeholder: string;
    value: string;
    onChange: (value: string) => void;
    status: "ok" | "missing" | "optional";
    helper?: string;
};

function Field({ label, placeholder, value, onChange, status, helper }: FieldProps) {
    const statusText = status === "ok" ? "Optional (already set)" : status === "optional" ? "Optional" : "Required";
    const statusColor = status === "ok" ? "text-emerald-300" : status === "optional" ? "text-blue-300" : "text-amber-200";

    return (
        <label className="block space-y-2">
            <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-white">{label}</span>
                <span className={`text-[11px] uppercase tracking-[0.18em] ${statusColor}`}>
                    {statusText}
                </span>
            </div>
            <input
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2.5 text-sm text-white placeholder:text-white/30 focus:border-white/50 focus:outline-none"
            />
            {helper && <p className="text-xs text-white/50">{helper}</p>}
        </label>
    );
}

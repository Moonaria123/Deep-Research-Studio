import { useEffect, useRef, useState } from "react";
import { Archive, Brain, ExternalLink, GitMerge, Pencil, Pin, PinOff, Plus, RefreshCw, RotateCcw, Settings, Square, Trash2, TrendingDown } from "lucide-react";
import { Link, useInRouterContext, useParams } from "react-router-dom";
import type {
  MemoryDreamSchedule,
  MemoryDreamStatus,
  MemoryBatchExpiryResult,
  MemoryItem,
  MemoryOptimizationGroup,
  MemorySettings,
  MemoryWriteAudit,
  WorkspaceMemoryRefreshStrategy,
} from "@/lib/types";
import { api, ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";
import { IS_DEMO } from "@/lib/runtime";
import { useGsapEntrance } from "@/hooks/useGsapEntrance";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/input";
import { toast } from "@/components/ui/toast";
import { SessionMemoryPanel } from "@/components/memory/SessionMemoryPanel";

const LAYERS: { key: string; label: string }[] = [
  { key: "session", label: "Session" },
  { key: "workspace", label: "Workspace" },
  { key: "user", label: "User" },
];

// The ranked list and manual add flow remain workspace/user-only. Session-layer
// rows use a separate management panel and never pass through Recall.
const ADD_LAYERS = LAYERS.filter((l) => l.key !== "session");
const FILTER_LAYERS = ADD_LAYERS;
type MemoryOptimizationSuggestion = MemoryOptimizationGroup["suggestions"][number];
type UserProfileSection = { id?: string; title: string; body: string };
type BatchExpiryNamespace = "workspace" | "user";
type BatchExpiryPreset = "never" | "30" | "90" | "custom" | "clear";
type BatchExpiryDraft = { preset: BatchExpiryPreset; date: string };
type BatchExpiryConfirm = {
  namespace: BatchExpiryNamespace;
  label: string;
  expiresAt: number | null;
  preview: MemoryBatchExpiryResult;
};
const DEFAULT_DREAM_STATUS: MemoryDreamStatus = {
  status: "idle",
  dream_run_id: null,
  workspace_key: "default",
  user_key: "default",
  started_at: null,
  finished_at: null,
  memories_created: 0,
  error: null,
  failure_count: 0,
  schedule: "off",
  next_scheduled_at: null,
};
const DEFAULT_MEMORY_SETTINGS: MemorySettings = {
  post_turn_memory_refresh_enabled: true,
  workspace_memory_refresh_strategy: "incremental",
  memory_thinking_enabled: false,
  format_probe: null,
};
const DEFAULT_WRITE_AUDIT: MemoryWriteAudit = {
  status: "idle",
  source: "none",
  workspace_key: "default",
  user_key: "default",
  attempted: 0,
  succeeded: 0,
  skipped: 0,
  failed: 0,
  error_code: null,
  checked_at: null,
  provider_ref: "",
  stale: false,
  incomplete_reasons: [],
};
const DEFAULT_BATCH_EXPIRY: BatchExpiryDraft = { preset: "never", date: "" };

function dreamVariant(status: MemoryDreamStatus["status"]) {
  if (status === "completed") return "success";
  if (["queued", "running", "cancelling"].includes(status)) return "warning";
  if (status === "error") return "danger";
  return "default";
}

function dreamLabel(status: MemoryDreamStatus["status"]) {
  if (status === "completed") return "Ready";
  if (status === "queued") return "Queued";
  if (status === "running") return "Running";
  if (status === "cancelling") return "Cancelling";
  if (status === "cancelled") return "Cancelled";
  if (status === "error") return "Failed";
  return "Idle";
}

function dreamStageLabel(stage?: string | null) {
  if (stage === "session_distill") return "Session summary";
  if (stage === "user_profile") return "User profile";
  if (stage === "unknown") return "Unknown stage";
  return stage || "";
}

function dreamIssueMessage(status: MemoryDreamStatus) {
  const parts: string[] = [];
  const failedCount = Number(status.failure_count ?? 0);
  if (status.error) parts.push(status.error);
  else if (failedCount > 0) parts.push(`${failedCount} refresh item${failedCount === 1 ? "" : "s"} skipped; existing memory was kept.`);
  else if (status.incomplete) parts.push("Not fully recorded; existing memory was kept.");
  if (status.error_code) parts.push(`Code: ${status.error_code}`);
  const stage = dreamStageLabel(status.failed_stage);
  if (stage) parts.push(`Stage: ${stage}`);
  if (status.failed_session_id) parts.push(`Session: ${status.failed_session_id}`);
  if (status.validation_error) parts.push(`Validation: ${status.validation_error}`);
  if (status.response_shape) parts.push(`Shape: ${status.response_shape}`);
  if (status.structured_strategy) parts.push(`Strategy: ${status.structured_strategy}`);
  if (typeof status.attempted_sessions === "number" || typeof status.failed_sessions === "number") {
    const attempted = status.attempted_sessions ?? 0;
    const failed = status.failed_sessions ?? 0;
    parts.push(`Sessions: ${attempted - failed}/${attempted} succeeded`);
  }
  if (status.user_profile_failed) parts.push("User profile skipped");
  return parts.join(" · ");
}

function writeAuditIssueMessage(audit: MemoryWriteAudit) {
  const parts: string[] = [];
  if (audit.stale) parts.push("Memory format test is stale");
  if (audit.failed > 0) parts.push(`${audit.failed} write item${audit.failed === 1 ? "" : "s"} failed`);
  if (audit.error_code) parts.push(`Code: ${audit.error_code}`);
  if (audit.incomplete_reasons?.length) parts.push(`Incomplete: ${audit.incomplete_reasons.join(", ")}`);
  return parts.join(" · ");
}

function formatTimestamp(value?: number | null) {
  if (!value) return "Never";
  const date = new Date(value * 1000);
  if (Number.isNaN(date.getTime())) return "Never";
  return date.toLocaleString();
}

function expiryLabel(value?: number | null) {
  if (!value) return "Never expires";
  const date = new Date(value * 1000);
  if (Number.isNaN(date.getTime())) return "Never expires";
  return `Expires ${date.toLocaleDateString()}`;
}

function expiryDateValue(value?: number | null) {
  if (!value) return "";
  const date = new Date(value * 1000);
  if (Number.isNaN(date.getTime())) return "";
  return date.toISOString().slice(0, 10);
}

function expiryFromDays(days: number) {
  return Math.floor((Date.now() + days * 24 * 60 * 60 * 1000) / 1000);
}

function expiryFromDate(value: string) {
  const date = new Date(`${value}T23:59:59`);
  return Number.isNaN(date.getTime()) ? null : Math.floor(date.getTime() / 1000);
}

function optimizerActionLabel(type: MemoryOptimizationSuggestion["type"]) {
  if (type === "archive") return "Archive duplicates";
  if (type === "downrank") return "Lower priority";
  return "Merge";
}

function optimizerActionSuccess(type: MemoryOptimizationSuggestion["type"]) {
  if (type === "archive") return "Archived duplicate memories.";
  if (type === "downrank") return "Lowered duplicate memory priority.";
  return "Merged duplicate memories.";
}

function optimizerActionError(type: MemoryOptimizationSuggestion["type"]) {
  if (type === "archive") return "Could not archive duplicate memories.";
  if (type === "downrank") return "Could not lower memory priority.";
  return "Could not merge memories.";
}

function memorySource(memory: MemoryItem) {
  return typeof memory.metadata?.source === "string" ? memory.metadata.source : "";
}

function sessionSummaryLabel(memory: MemoryItem) {
  const title = typeof memory.metadata?.session_title === "string" ? memory.metadata.session_title.trim() : "";
  const sessionId = typeof memory.metadata?.session_id === "string" ? memory.metadata.session_id.trim() : "";
  return `Session summary · ${title || sessionId || "Untitled session"}`;
}

function sessionSummaryHref(memory: MemoryItem) {
  const sessionId = typeof memory.metadata?.session_id === "string" ? memory.metadata.session_id.trim() : "";
  return sessionId ? `/session/${encodeURIComponent(sessionId)}` : "";
}

function SourceSessionLink({ href, memoryId }: { href: string; memoryId: string }) {
  const inRouter = useInRouterContext();
  const className = "inline-flex items-center gap-1 text-accent underline-offset-2 hover:underline focus-ring";
  const label = `Open source session for memory ${memoryId}`;
  const content = (
    <>
      <ExternalLink className="h-3 w-3" aria-hidden />
      Open session
    </>
  );
  if (inRouter) {
    return (
      <Link to={href} className={className} aria-label={label}>
        {content}
      </Link>
    );
  }
  return (
    <a href={href} className={className} aria-label={label}>
      {content}
    </a>
  );
}

function userProfileSections(memory: MemoryItem): UserProfileSection[] {
  const raw = memory.metadata?.sections;
  if (!Array.isArray(raw)) return [];
  return raw.reduce<UserProfileSection[]>((sections, item) => {
    if (!item || typeof item !== "object") return sections;
      const section = item as Record<string, unknown>;
      const title = typeof section.title === "string" ? section.title.trim() : "";
      const body = typeof section.body === "string" ? section.body.trim() : "";
      const id = typeof section.id === "string" ? section.id : undefined;
    if (title && body) sections.push({ id, title, body });
    return sections;
  }, []);
}

function memoryMutationError(error: unknown, fallback: string) {
  if (error instanceof ApiError && error.code === "MEMORY_PINNED") {
    return "Pinned memories cannot receive a TTL or be archived. Unpin the memory first.";
  }
  return error instanceof ApiError ? error.message : fallback;
}

export function Memory() {
  const { memoryId: focusedMemoryId } = useParams<{ memoryId: string }>();
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [archivedItems, setArchivedItems] = useState<MemoryItem[]>([]);
  const [dreamStatus, setDreamStatus] = useState<MemoryDreamStatus>(DEFAULT_DREAM_STATUS);
  const [layer, setLayer] = useState<string>("all");
  const [draft, setDraft] = useState("");
  const [draftLayer, setDraftLayer] = useState("workspace");
  const [adding, setAdding] = useState(false);
  const [dreaming, setDreaming] = useState(false);
  const [dreamCancelling, setDreamCancelling] = useState(false);
  const [scheduleBusy, setScheduleBusy] = useState(false);
  const [archivedOpen, setArchivedOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<MemoryItem | null>(null);
  const [editTarget, setEditTarget] = useState<MemoryItem | null>(null);
  const [editContent, setEditContent] = useState("");
  const [editSalience, setEditSalience] = useState("");
  const [optimizeOpen, setOptimizeOpen] = useState(false);
  const [optimizeGroups, setOptimizeGroups] = useState<MemoryOptimizationGroup[]>([]);
  const [optimizeBusy, setOptimizeBusy] = useState(false);
  const [optimizeError, setOptimizeError] = useState("");
  const [restoreWarning, setRestoreWarning] = useState("");
  const [memoryRoleConfigured, setMemoryRoleConfigured] = useState(false);
  const [memorySettings, setMemorySettings] = useState<MemorySettings>(DEFAULT_MEMORY_SETTINGS);
  const [writeAudit, setWriteAudit] = useState<MemoryWriteAudit>(DEFAULT_WRITE_AUDIT);
  const [memorySettingsBusy, setMemorySettingsBusy] = useState(false);
  const [batchExpiry, setBatchExpiry] = useState<Record<BatchExpiryNamespace, BatchExpiryDraft>>({
    workspace: DEFAULT_BATCH_EXPIRY,
    user: DEFAULT_BATCH_EXPIRY,
  });
  const [batchExpiryBusy, setBatchExpiryBusy] = useState<BatchExpiryNamespace | null>(null);
  const [batchExpiryConfirm, setBatchExpiryConfirm] = useState<BatchExpiryConfirm | null>(null);
  const [busyActions, setBusyActions] = useState<string[]>([]);
  const busyRef = useRef<Set<string>>(new Set());
  const ref = useGsapEntrance<HTMLDivElement>([items.length, layer]);

  const loadMemories = () =>
    api
      .recallMemory({ workspace_key: "default", user_key: "default", limit: 100 })
      .then((r) => setItems(r.memories))
      .catch(() => undefined);

  const loadDreamStatus = () =>
    api
      .getMemoryDreamStatus({ workspace_key: "default", user_key: "default" })
      .then((status) => {
        setDreamStatus(status);
        return status;
      })
      .catch(() => undefined);

  const loadArchived = () =>
    api
      .listArchivedMemories({ workspace_key: "default", user_key: "default", limit: 100 })
      .then((r) => setArchivedItems(r.memories))
      .catch(() => undefined);

  const loadMemorySettings = () =>
    api
      .getMemorySettings()
      .then(setMemorySettings)
      .catch(() => undefined);

  const loadWriteAudit = () =>
    api
      .getMemoryWriteAudit({ workspace_key: "default", user_key: "default" })
      .then(setWriteAudit)
      .catch(() => undefined);

  const load = () => {
    void loadMemories();
    void loadDreamStatus();
    void loadArchived();
    void loadMemorySettings();
    void loadWriteAudit();
  };

  const showLibraryIndexWarning = (warning?: { message?: string } | null) => {
    const message = warning?.message ?? "";
    setRestoreWarning(message);
    if (message) toast.info(message);
  };

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    if (!focusedMemoryId) return undefined;
    let cancelled = false;
    api
      .getMemory(focusedMemoryId)
      .then(({ memory }) => {
        if (cancelled) return;
        setLayer("all");
        if (memory.archived_at != null) {
          setArchivedItems((current) => [memory, ...current.filter((item) => item.id !== memory.id)]);
          setArchivedOpen(true);
        } else {
          setItems((current) => [memory, ...current.filter((item) => item.id !== memory.id)]);
        }
      })
      .catch((error) => {
        if (!cancelled) toast.error(error instanceof ApiError ? error.message : "Could not open cited memory.");
      });
    return () => {
      cancelled = true;
    };
  }, [focusedMemoryId]);

  useEffect(() => {
    if (!focusedMemoryId) return undefined;
    const timeout = window.setTimeout(() => {
      document.getElementById(`memory-${focusedMemoryId}`)?.scrollIntoView?.({ behavior: "smooth", block: "center" });
    }, 0);
    return () => window.clearTimeout(timeout);
  }, [archivedItems, archivedOpen, focusedMemoryId, items]);

  useEffect(() => {
    if (!["queued", "running", "cancelling"].includes(dreamStatus.status)) return undefined;
    const id = window.setInterval(() => {
      void loadDreamStatus().then((status) => {
        if (status && ["completed", "cancelled", "error"].includes(status.status)) {
          void loadMemories();
          void loadWriteAudit();
        }
      });
    }, 1500);
    return () => window.clearInterval(id);
  }, [dreamStatus.status]);

  const beginAction = (key: string) => {
    if (busyRef.current.has(key)) return false;
    busyRef.current.add(key);
    setBusyActions([...busyRef.current]);
    return true;
  };

  const endAction = (key: string) => {
    busyRef.current.delete(key);
    setBusyActions([...busyRef.current]);
  };

  const isActionBusy = (key: string) => busyActions.includes(key);

  const addMemory = async (e: React.FormEvent) => {
    e.preventDefault();
    const content = draft.trim();
    if (!content || adding) return;
      setAdding(true);
    try {
      await api.addMemory({ layer: draftLayer, scope_key: "default", content });
      setDraft("");
      await loadMemories();
    } catch {
      // Leave the draft so the user can retry.
    } finally {
      setAdding(false);
    }
  };

  const refreshDream = async () => {
    if (dreaming || ["queued", "running", "cancelling"].includes(dreamStatus.status)) return;
    setDreaming(true);
    try {
      const status = await api.startMemoryDream({ workspace_key: "default", user_key: "default" });
      setDreamStatus(status);
      if (status.status === "completed") await loadMemories();
      await loadWriteAudit();
    } catch {
      await loadDreamStatus();
    } finally {
      setDreaming(false);
    }
  };

  const cancelDream = async () => {
    if (!dreamStatus.dream_run_id || dreamCancelling || dreamStatus.status === "cancelling") return;
    setDreamCancelling(true);
    try {
      const status = await api.cancelMemoryDream(dreamStatus.dream_run_id);
      setDreamStatus(status);
    } catch {
      await loadDreamStatus();
    } finally {
      setDreamCancelling(false);
    }
  };

  const updateDreamSchedule = async (value: MemoryDreamSchedule) => {
    setScheduleBusy(true);
    try {
      const next = await api.updateMemoryDreamSchedule({ schedule: value });
      setDreamStatus((current) => ({
        ...current,
        schedule: next.schedule,
        next_scheduled_at: next.next_scheduled_at ?? null,
      }));
    } finally {
      setScheduleBusy(false);
    }
  };

  const updateMemorySettings = async (patch: Partial<MemorySettings>) => {
    setMemorySettingsBusy(true);
    try {
      const next = await api.updateMemorySettings(patch);
      setMemorySettings(next);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update memory settings.");
    } finally {
      setMemorySettingsBusy(false);
    }
  };

  const archiveMemory = async (memory: MemoryItem) => {
    const key = `archive:${memory.id}`;
    if (!beginAction(key)) return;
    try {
      await api.archiveMemory(memory.id);
      toast.success("Archived memory.");
      await Promise.all([loadMemories(), loadArchived()]);
    } catch (err) {
      toast.error(memoryMutationError(err, "Could not archive memory."));
    } finally {
      endAction(key);
    }
  };

  const restoreMemory = async (memory: MemoryItem) => {
    const key = `restore:${memory.id}`;
    if (!beginAction(key)) return;
    try {
      const result = await api.restoreMemory(memory.id);
      toast.success("Restored memory.");
      showLibraryIndexWarning(result.library_index_warning);
      await Promise.all([loadMemories(), loadArchived()]);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not restore memory.");
    } finally {
      endAction(key);
    }
  };

  const openEditMemory = (memory: MemoryItem) => {
    setEditTarget(memory);
    setEditContent(memory.content);
    setEditSalience(String(memory.salience));
  };

  const saveEditedMemory = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!editTarget) return;
    const content = editContent.trim();
    const salience = Number.parseFloat(editSalience);
    if (!content || !Number.isFinite(salience)) return;
    const key = `edit:${editTarget.id}`;
    if (!beginAction(key)) return;
    try {
      const result = await api.patchMemory(editTarget.id, { content, salience });
      toast.success("Updated memory.");
      showLibraryIndexWarning(result.library_index_warning);
      setEditTarget(null);
      await loadMemories();
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update memory.");
    } finally {
      endAction(key);
    }
  };

  const updateMemoryPin = async (memory: MemoryItem) => {
    const key = `pin:${memory.id}`;
    if (!beginAction(key)) return;
    try {
      const result = await api.patchMemory(memory.id, { pinned: !memory.pinned });
      setItems((current) => current.map((item) => (item.id === memory.id ? result.memory : item)));
      toast.success(result.memory.pinned ? "Pinned memory." : "Unpinned memory.");
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not update memory pin.");
    } finally {
      endAction(key);
    }
  };

  const updateMemoryExpiry = async (memory: MemoryItem, expiresAt: number | null) => {
    const key = `ttl:${memory.id}`;
    if (!beginAction(key)) return;
    try {
      const result = await api.patchMemory(memory.id, { expires_at: expiresAt });
      setItems((current) => current.map((item) => (item.id === memory.id ? result.memory : item)));
      toast.success(expiresAt ? "Memory expiry updated." : "Memory expiry cleared.");
    } catch (err) {
      toast.error(memoryMutationError(err, "Could not update memory expiry."));
    } finally {
      endAction(key);
    }
  };

  const applyTtlPreset = async (memory: MemoryItem, value: string) => {
    if (value === "never") {
      await updateMemoryExpiry(memory, null);
    } else if (value === "30") {
      await updateMemoryExpiry(memory, expiryFromDays(30));
    } else if (value === "90") {
      await updateMemoryExpiry(memory, expiryFromDays(90));
    }
  };

  const updateBatchExpiry = (namespace: BatchExpiryNamespace, patch: Partial<BatchExpiryDraft>) => {
    setBatchExpiry((current) => ({
      ...current,
      [namespace]: { ...current[namespace], ...patch },
    }));
  };

  const batchExpiryValue = (draft: BatchExpiryDraft) => {
    if (draft.preset === "30") return expiryFromDays(30);
    if (draft.preset === "90") return expiryFromDays(90);
    if (draft.preset === "custom") return draft.date ? expiryFromDate(draft.date) : null;
    return null;
  };

  const previewBatchExpiry = async (namespace: BatchExpiryNamespace, label: string) => {
    const draft = batchExpiry[namespace];
    const expiresAt = batchExpiryValue(draft);
    if (draft.preset === "custom" && !expiresAt) {
      toast.error("Choose a valid custom expiry date.");
      return;
    }
    setBatchExpiryBusy(namespace);
    try {
      const preview = await api.batchMemoryExpiry({
        workspace_key: "default",
        user_key: "default",
        namespace,
        expires_at: expiresAt,
        dry_run: true,
      });
      setBatchExpiryConfirm({ namespace, label, expiresAt, preview });
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not preview memory expiry.");
    } finally {
      setBatchExpiryBusy(null);
    }
  };

  const confirmBatchExpiry = async () => {
    if (!batchExpiryConfirm) return;
    const { namespace, expiresAt } = batchExpiryConfirm;
    setBatchExpiryBusy(namespace);
    try {
      const result = await api.batchMemoryExpiry({
        workspace_key: "default",
        user_key: "default",
        namespace,
        expires_at: expiresAt,
        dry_run: false,
      });
      toast.success(
        `Updated ${result.updated} ${namespace} memories; skipped ${result.skipped_pinned} pinned and ${result.skipped_archived} archived.`,
      );
      setBatchExpiryConfirm(null);
      await Promise.all([loadMemories(), loadArchived()]);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not apply memory expiry.");
    } finally {
      setBatchExpiryBusy(null);
    }
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    const key = `delete:${deleteTarget.id}`;
    if (!beginAction(key)) return;
    try {
      await api.deleteMemory(deleteTarget.id);
      toast.success("Deleted memory.");
      await Promise.all([loadMemories(), loadArchived()]);
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : "Could not delete memory.");
    } finally {
      endAction(key);
      setDeleteTarget(null);
    }
  };

  const refreshOptimizationPreview = async () => {
    const preview = await api.previewMemoryOptimization({ workspace_key: "default", user_key: "default" });
    setOptimizeGroups(preview.groups);
    return preview;
  };

  const openOptimize = async () => {
    setOptimizeOpen(true);
    setOptimizeBusy(true);
    setOptimizeError("");
    try {
      const [preview, registry] = await Promise.all([
        api.previewMemoryOptimization({ workspace_key: "default", user_key: "default" }),
        api.llmRegistry().catch(() => null),
      ]);
      setOptimizeGroups(preview.groups);
      setMemoryRoleConfigured(Boolean(registry?.roles?.memory?.trim()));
    } catch (err) {
      setOptimizeGroups([]);
      setOptimizeError(err instanceof ApiError ? err.message : "Could not review memories.");
    } finally {
      setOptimizeBusy(false);
    }
  };

  const applyOptimizerAction = async (
    group: MemoryOptimizationGroup,
    suggestion: MemoryOptimizationSuggestion,
  ) => {
    if (suggestion.type === "merge" && !memoryRoleConfigured) return;
    const key = `optimize:${group.group_id}:${suggestion.type}`;
    if (!beginAction(key)) return;
    setOptimizeError("");
    try {
      const memoryIds = suggestion.memory_ids.length ? suggestion.memory_ids : group.items.map((item) => item.id);
      const result = await api.applyMemoryOptimization({
        strategy: suggestion.type,
        memory_ids: memoryIds,
        ...(suggestion.type === "downrank" ? { factor: suggestion.factor ?? 0.5 } : {}),
      });
      toast.success(optimizerActionSuccess(suggestion.type));
      if (suggestion.type === "merge") showLibraryIndexWarning(result.library_index_warning);
      await Promise.all([loadMemories(), loadArchived(), refreshOptimizationPreview()]);
    } catch (err) {
      setOptimizeError(err instanceof ApiError ? err.message : optimizerActionError(suggestion.type));
    } finally {
      endAction(key);
    }
  };

  const filtered = layer === "all" ? items : items.filter((i) => i.layer === layer);
  const lastRefresh = dreamStatus.finished_at ?? dreamStatus.started_at;
  const nextSchedule = dreamStatus.next_scheduled_at;
  const isDreamRunning = dreaming || ["queued", "running", "cancelling"].includes(dreamStatus.status);
  const dreamIssue = dreamIssueMessage(dreamStatus);
  const writeAuditIssue = writeAuditIssueMessage(writeAudit);
  const deleteBusy = deleteTarget ? isActionBusy(`delete:${deleteTarget.id}`) : false;
  const editBusy = editTarget ? isActionBusy(`edit:${editTarget.id}`) : false;
  const renderBatchExpiryControl = (namespace: BatchExpiryNamespace, label: string) => {
    const draft = batchExpiry[namespace];
    const busy = batchExpiryBusy === namespace;
    return (
      <div className="rounded-sm bg-bg-subtle px-3 py-2">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <span className="font-medium text-text-body">{label}</span>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={IS_DEMO || busy || (draft.preset === "custom" && !draft.date)}
            onClick={() => void previewBatchExpiry(namespace, label)}
          >
            {busy ? "Checking" : `Apply ${namespace}`}
          </Button>
        </div>
        <div className="flex flex-wrap gap-2">
          <select
            aria-label={`${label} preset`}
            value={draft.preset}
            disabled={IS_DEMO || busy}
            onChange={(event) => updateBatchExpiry(namespace, { preset: event.target.value as BatchExpiryPreset })}
            className="h-8 rounded-sm border border-border bg-surface px-2 text-xs text-text-body focus-ring"
          >
            <option value="never">No expiry</option>
            <option value="30">30 days</option>
            <option value="90">90 days</option>
            <option value="custom">Custom date</option>
            <option value="clear">Clear expiry</option>
          </select>
          {draft.preset === "custom" ? (
            <input
              aria-label={`${label} custom date`}
              type="date"
              value={draft.date}
              disabled={IS_DEMO || busy}
              onChange={(event) => updateBatchExpiry(namespace, { date: event.target.value })}
              className="h-8 rounded-sm border border-border bg-surface px-2 text-xs text-text-body focus-ring"
            />
          ) : null}
        </div>
      </div>
    );
  };

  return (
    <div ref={ref} className="mx-auto w-full max-w-[820px] px-6 py-8">
      <div data-anim className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="mb-1 text-2xl font-bold text-text">Memory</h1>
          <p className="text-sm text-text-muted">Cross-session knowledge, ranked by salience and recency. Stored locally.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="outline" size="sm" onClick={() => setArchivedOpen(true)}>
            <Archive className="h-4 w-4" /> Archived
          </Button>
          <Button type="button" variant="outline" size="sm" onClick={() => void openOptimize()} disabled={optimizeBusy}>
            <GitMerge className="h-4 w-4" /> Review duplicates
          </Button>
        </div>
      </div>

      {restoreWarning ? (
        <p className="mb-5 rounded-sm border border-warning/50 bg-warning/10 px-3 py-2 text-sm text-text-body" role="status">
          {restoreWarning}
        </p>
      ) : null}

      <Card data-anim className="mb-5">
        <CardContent className="pt-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold text-text">Research profile</h2>
              <div className="mt-3 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
                <div>
                  <span className="block text-text-faint">Last refresh</span>
                  <span className="mt-1 block text-text-body">{formatTimestamp(lastRefresh)}</span>
                </div>
                <div>
                  <span className="block text-text-faint">New memories</span>
                  <span className="mt-1 block text-text-body">{dreamStatus.memories_created}</span>
                </div>
                <div>
                  <span className="block text-text-faint">Workspace</span>
                  <span className="mt-1 block text-text-body">{dreamStatus.workspace_key}</span>
                </div>
                <div>
                  <span className="block text-text-faint">Next schedule</span>
                  <span className="mt-1 block text-text-body">{formatTimestamp(nextSchedule)}</span>
                </div>
                <div>
                  <span className="block text-text-faint">Workspace delta</span>
                  <span className="mt-1 block text-text-body">{dreamStatus.workspace_skipped ?? 0} skipped</span>
                </div>
                <div>
                  <span className="block text-text-faint">Format</span>
                  <span className="mt-1 block text-text-body">{dreamStatus.structured_strategy || "Unknown"}</span>
                </div>
                <div>
                  <span className="block text-text-faint">Write audit</span>
                  <span className="mt-1 block text-text-body">
                    {writeAudit.succeeded}/{writeAudit.attempted} succeeded
                  </span>
                </div>
                <div>
                  <span className="block text-text-faint">Skipped</span>
                  <span className="mt-1 block text-text-body">{writeAudit.skipped}</span>
                </div>
              </div>
            </div>
            <div className="flex flex-wrap items-center justify-end gap-2">
              <select
                aria-label="Dreaming schedule"
                value={dreamStatus.schedule ?? "off"}
                disabled={IS_DEMO || scheduleBusy}
                onChange={(e) => void updateDreamSchedule(e.target.value as MemoryDreamSchedule)}
                className="h-8 rounded-sm border border-border bg-surface px-2 text-xs text-text-body focus-ring"
              >
                <option value="off">Off</option>
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
              </select>
              <Badge variant={dreamVariant(dreamStatus.status)} aria-live="polite">
                {dreamLabel(dreamStatus.status)}
              </Badge>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={refreshDream}
                disabled={IS_DEMO || isDreamRunning}
              >
                <RefreshCw className={cn("h-4 w-4", isDreamRunning && "animate-spin")} />
                {isDreamRunning ? "Refreshing" : "Refresh"}
              </Button>
              {dreamStatus.dream_run_id && ["queued", "running", "cancelling"].includes(dreamStatus.status) ? (
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => void cancelDream()}
                  disabled={IS_DEMO || dreamCancelling || dreamStatus.status === "cancelling"}
                >
                  <Square className="h-3.5 w-3.5" />
                  {dreamStatus.status === "cancelling" || dreamCancelling ? "Cancelling" : "Cancel"}
                </Button>
              ) : null}
            </div>
          </div>
          <div className="mt-4 grid gap-3 border-t border-border pt-4 text-xs sm:grid-cols-2">
            <label className="flex items-center justify-between gap-3 rounded-sm bg-bg-subtle px-3 py-2 text-text-muted">
              <span>Post-turn refresh</span>
              <span className="flex items-center gap-2">
                <span className="text-text-body">
                  {memorySettings.post_turn_memory_refresh_enabled ? "On" : "Off"}
                </span>
                <Switch
                  checked={memorySettings.post_turn_memory_refresh_enabled}
                  disabled={IS_DEMO || memorySettingsBusy}
                  aria-label="Post-turn memory refresh"
                  onCheckedChange={(checked) =>
                    void updateMemorySettings({ post_turn_memory_refresh_enabled: checked })
                  }
                />
              </span>
            </label>
            <label className="flex items-center justify-between gap-3 rounded-sm bg-bg-subtle px-3 py-2 text-text-muted">
              <span>Workspace strategy</span>
              <select
                aria-label="Workspace memory refresh strategy"
                value={memorySettings.workspace_memory_refresh_strategy}
                disabled={IS_DEMO || memorySettingsBusy}
                onChange={(event) =>
                  void updateMemorySettings({
                    workspace_memory_refresh_strategy: event.target.value as WorkspaceMemoryRefreshStrategy,
                  })
                }
                className="h-8 rounded-sm border border-border bg-surface px-2 text-xs text-text-body focus-ring"
              >
                <option value="incremental">Incremental</option>
                <option value="full">Full</option>
              </select>
            </label>
          </div>
          <div className="mt-3 grid gap-3 border-t border-border pt-4 text-xs sm:grid-cols-2">
            {renderBatchExpiryControl("workspace", "Workspace memory expiry")}
            {renderBatchExpiryControl("user", "User memory expiry")}
          </div>
          {dreamIssue ? (
            <p
              className={cn(
                "mt-3 rounded-sm border px-3 py-2 text-xs",
                dreamStatus.status === "error"
                  ? "border-danger/50 bg-danger/10 text-danger"
                  : "border-warning/50 bg-warning/10 text-text-body",
              )}
              role="status"
            >
              {dreamIssue}
            </p>
          ) : null}
          {writeAuditIssue ? (
            <p className="mt-3 rounded-sm border border-warning/50 bg-warning/10 px-3 py-2 text-xs text-text-body" role="status">
              {writeAuditIssue}
            </p>
          ) : null}
        </CardContent>
      </Card>

      <SessionMemoryPanel />

      <Card data-anim className="mb-5">
        <CardContent className="pt-5">
          <form onSubmit={addMemory} className="space-y-3">
            <Textarea
              aria-label="New memory content"
              placeholder="Add a fact, preference, or constraint to remember…"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              rows={2}
              disabled={IS_DEMO}
            />
            <div className="flex items-center gap-2">
              <select
                aria-label="Memory layer"
                value={draftLayer}
                disabled={IS_DEMO}
                onChange={(e) => setDraftLayer(e.target.value)}
                className="h-9 rounded-sm border border-border bg-surface px-2 text-sm text-text-body focus-ring"
              >
                {ADD_LAYERS.map((l) => (
                  <option key={l.key} value={l.key}>
                    {l.label}
                  </option>
                ))}
              </select>
              <Button type="submit" size="sm" disabled={IS_DEMO || adding || !draft.trim()}>
                <Plus className="h-4 w-4" /> {adding ? "Adding…" : "Add memory"}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      <div data-anim className="mb-4 flex gap-2">
        {[{ key: "all", label: "All" }, ...FILTER_LAYERS].map((l) => (
          <button
            key={l.key}
            onClick={() => setLayer(l.key)}
            aria-pressed={layer === l.key}
            className={cn(
              "rounded-full border px-3 py-1 text-xs font-medium transition-colors focus-ring",
              layer === l.key ? "border-accent bg-accent-soft text-accent" : "border-border text-text-muted hover:text-text-body",
            )}
          >
            {l.label}
          </button>
        ))}
      </div>

      {filtered.length === 0 ? (
        <Card data-anim>
          <CardContent className="py-10 text-center text-sm text-text-muted">No memories yet.</CardContent>
        </Card>
      ) : (
        <ul className="space-y-2">
          {filtered.map((m) => (
            <li
              key={m.id}
              id={`memory-${m.id}`}
              data-anim
              data-memory-id={m.id}
              data-memory-focused={focusedMemoryId === m.id ? "true" : undefined}
            >
              <Card className={focusedMemoryId === m.id ? "border-accent ring-1 ring-accent" : undefined}>
                <CardContent className="flex items-start gap-3 py-3">
                  <Brain className="mt-0.5 h-4 w-4 shrink-0 text-text-muted" aria-hidden />
                  <div className="min-w-0 flex-1 text-sm text-text-body">
                    {memorySource(m) === "session_distill_v1" ? (
                      <div className="mb-1 flex flex-wrap items-center gap-2 text-xs font-medium text-text-muted">
                        <span>{sessionSummaryLabel(m)}</span>
                        {sessionSummaryHref(m) ? (
                          <SourceSessionLink href={sessionSummaryHref(m)} memoryId={m.id} />
                        ) : null}
                      </div>
                    ) : null}
                    {memorySource(m) === "user_profile_v1" ? (
                      <p className="mb-2 text-xs font-medium text-text-muted">User profile</p>
                    ) : null}
                    {userProfileSections(m).length > 0 ? (
                      <div className="space-y-2">
                        {userProfileSections(m).map((section, index) => (
                          <section key={section.id ?? `${m.id}-${index}`} className="rounded-sm bg-bg-subtle px-3 py-2">
                            <h3 className="text-xs font-semibold text-text">{section.title}</h3>
                            <p className="mt-1 whitespace-pre-wrap text-sm text-text-body">{section.body}</p>
                          </section>
                        ))}
                      </div>
                    ) : (
                      <p className="whitespace-pre-wrap">{m.content}</p>
                    )}
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-1">
                    <div className="flex flex-wrap justify-end gap-1">
                      {m.pinned ? <Badge variant="success">Pinned</Badge> : null}
                      <Badge variant="accent">{m.layer}</Badge>
                    </div>
                    <span className="text-[11px] text-text-faint">
                      salience {(m.effective_salience ?? m.salience).toFixed(2)}
                    </span>
                    {m.layer === "workspace" || m.layer === "user" ? (
                      <>
                        <span className="text-[11px] text-text-faint">{expiryLabel(m.expires_at)}</span>
                        <div className="flex flex-wrap justify-end gap-1">
                          <select
                            aria-label={`Memory expiry preset ${m.id}`}
                            className="h-8 rounded-sm border border-border bg-surface px-2 text-[11px] text-text-body focus-ring"
                            defaultValue="never"
                            disabled={IS_DEMO || m.pinned || isActionBusy(`ttl:${m.id}`)}
                            onChange={(event) => void applyTtlPreset(m, event.target.value)}
                          >
                            <option value="never">No expiry</option>
                            <option value="30">30 days</option>
                            <option value="90">90 days</option>
                          </select>
                          <input
                            aria-label={`Custom expiry ${m.id}`}
                            type="date"
                            value={expiryDateValue(m.expires_at)}
                            disabled={IS_DEMO || m.pinned || isActionBusy(`ttl:${m.id}`)}
                            onChange={(event) => {
                              const next = event.target.value ? expiryFromDate(event.target.value) : null;
                              void updateMemoryExpiry(m, next);
                            }}
                            className="h-8 w-[126px] rounded-sm border border-border bg-surface px-2 text-[11px] text-text-body focus-ring"
                          />
                        </div>
                        <div className="mt-1 flex gap-1">
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            onClick={() => void updateMemoryPin(m)}
                            disabled={IS_DEMO || isActionBusy(`pin:${m.id}`)}
                            aria-label={`${m.pinned ? "Unpin" : "Pin"} memory ${m.id}`}
                          >
                            {m.pinned ? <PinOff className="h-4 w-4" /> : <Pin className="h-4 w-4" />}
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            onClick={() => openEditMemory(m)}
                            disabled={IS_DEMO || isActionBusy(`edit:${m.id}`)}
                            aria-label={`Edit memory ${m.id}`}
                          >
                            <Pencil className="h-4 w-4" />
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            onClick={() => void archiveMemory(m)}
                            disabled={IS_DEMO || m.pinned || isActionBusy(`archive:${m.id}`)}
                            aria-label={`Archive memory ${m.id}`}
                          >
                            <Archive className="h-4 w-4" />
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            onClick={() => setDeleteTarget(m)}
                            disabled={IS_DEMO}
                            aria-label={`Delete memory ${m.id}`}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </>
                    ) : null}
                  </div>
                </CardContent>
              </Card>
            </li>
          ))}
        </ul>
      )}

      <Dialog
        open={Boolean(batchExpiryConfirm)}
        onOpenChange={(open) => {
          if (!open) setBatchExpiryConfirm(null);
        }}
        title="Apply memory expiry"
        description="Pinned and archived memories will not be changed."
      >
        {batchExpiryConfirm ? (
          <div className="space-y-4 text-sm text-text-body">
            <p>
              Apply <span className="font-medium">{batchExpiryConfirm.label}</span> to{" "}
              <span className="font-medium">{batchExpiryConfirm.preview.would_update}</span> active memories?
            </p>
            <dl className="grid grid-cols-3 gap-2 rounded-sm bg-bg-subtle p-3 text-xs">
              <div>
                <dt className="text-text-faint">Will update</dt>
                <dd className="text-text-body">{batchExpiryConfirm.preview.would_update}</dd>
              </div>
              <div>
                <dt className="text-text-faint">Pinned skipped</dt>
                <dd className="text-text-body">{batchExpiryConfirm.preview.skipped_pinned}</dd>
              </div>
              <div>
                <dt className="text-text-faint">Archived skipped</dt>
                <dd className="text-text-body">{batchExpiryConfirm.preview.skipped_archived}</dd>
              </div>
            </dl>
            <div className="flex justify-end gap-2">
              <Button type="button" variant="ghost" onClick={() => setBatchExpiryConfirm(null)}>
                Cancel
              </Button>
              <Button
                type="button"
                onClick={() => void confirmBatchExpiry()}
                disabled={IS_DEMO || batchExpiryBusy === batchExpiryConfirm.namespace}
              >
                {batchExpiryBusy === batchExpiryConfirm.namespace ? "Applying" : "Apply expiry"}
              </Button>
            </div>
          </div>
        ) : null}
      </Dialog>

      <Dialog
        open={archivedOpen}
        onOpenChange={setArchivedOpen}
        title="Archived memories"
        description={`${archivedItems.length} archived`}
      >
        {archivedItems.length === 0 ? (
          <p className="text-sm text-text-muted">No archived memories.</p>
        ) : (
          <ul className="max-h-[420px] space-y-2 overflow-y-auto">
            {archivedItems.map((memory) => (
              <li
                key={memory.id}
                id={`memory-${memory.id}`}
                data-memory-id={memory.id}
                data-memory-focused={focusedMemoryId === memory.id ? "true" : undefined}
                className={cn(
                  "rounded-md border border-border p-3",
                  focusedMemoryId === memory.id && "border-accent ring-1 ring-accent",
                )}
              >
                <p className="text-sm text-text-body">{memory.content}</p>
                <div className="mt-2 flex items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="accent">{memory.layer}</Badge>
                    <span className="text-xs text-text-faint">Archived {formatTimestamp(memory.archived_at)}</span>
                  </div>
                  <div className="flex gap-1">
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => void restoreMemory(memory)}
                      disabled={IS_DEMO || isActionBusy(`restore:${memory.id}`)}
                      aria-label={`Restore memory ${memory.id}`}
                    >
                      <RotateCcw className="h-4 w-4" /> Restore
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="danger"
                      onClick={() => setDeleteTarget(memory)}
                      disabled={IS_DEMO || isActionBusy(`delete:${memory.id}`)}
                      aria-label={`Delete archived memory ${memory.id}`}
                    >
                      <Trash2 className="h-4 w-4" /> Delete
                    </Button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Dialog>

      <Dialog
        open={optimizeOpen}
        onOpenChange={setOptimizeOpen}
        title="Review duplicate memories"
        description={`${optimizeGroups.length} candidate groups`}
      >
        {optimizeError ? (
          <p className="mb-3 text-sm text-danger" role="status">
            {optimizeError}
          </p>
        ) : null}
        <p className="mb-3 text-xs text-text-muted">Pinned memories are protected and omitted from automatic archive, merge, or downrank suggestions.</p>
        {optimizeBusy && optimizeGroups.length === 0 ? (
          <p className="text-sm text-text-muted">Reviewing memories...</p>
        ) : optimizeGroups.length === 0 ? (
          <p className="text-sm text-text-muted">No duplicate candidates.</p>
        ) : (
          <ul className="max-h-[460px] space-y-3 overflow-y-auto">
            {optimizeGroups.map((group) => (
              <li key={group.group_id} className="rounded-md border border-border p-3">
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <Badge variant="default">similarity {group.similarity.toFixed(2)}</Badge>
                  <div className="flex flex-wrap justify-end gap-2">
                    {group.suggestions.map((suggestion) => {
                      const key = `optimize:${group.group_id}:${suggestion.type}`;
                      const mergeBlocked = suggestion.type === "merge" && !memoryRoleConfigured;
                      return (
                        <Button
                          key={`${suggestion.type}:${suggestion.memory_ids.join(",")}`}
                          type="button"
                          size="sm"
                          variant={suggestion.type === "merge" ? "default" : "outline"}
                          onClick={() => void applyOptimizerAction(group, suggestion)}
                          disabled={IS_DEMO || optimizeBusy || isActionBusy(key) || mergeBlocked}
                        >
                          {suggestion.type === "archive" ? <Archive className="h-4 w-4" /> : null}
                          {suggestion.type === "downrank" ? <TrendingDown className="h-4 w-4" /> : null}
                          {suggestion.type === "merge" ? <GitMerge className="h-4 w-4" /> : null}
                          {optimizerActionLabel(suggestion.type)}
                        </Button>
                      );
                    })}
                  </div>
                </div>
                {!memoryRoleConfigured && group.suggestions.some((suggestion) => suggestion.type === "merge") ? (
                  <a
                    href="/settings"
                    className="mb-2 inline-flex items-center gap-1 text-xs font-medium text-accent hover:underline focus-ring"
                  >
                    <Settings className="h-3.5 w-3.5" /> Set memory processing model
                  </a>
                ) : null}
                <p className="mb-2 text-sm text-text-body">{group.suggested_summary}</p>
                <ul className="space-y-1">
                  {group.items.map((item) => (
                    <li key={item.id} className="text-xs text-text-muted">
                      {item.content}
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        )}
      </Dialog>

      <Dialog
        open={Boolean(editTarget)}
        onOpenChange={(open) => !open && setEditTarget(null)}
        title="Edit memory"
        description="Update workspace or user memory content."
      >
        <form onSubmit={saveEditedMemory} className="space-y-3">
          <Textarea
            aria-label="Edit memory content"
            value={editContent}
            onChange={(event) => setEditContent(event.target.value)}
            rows={4}
            disabled={IS_DEMO}
          />
          <label className="block text-sm text-text-body">
            <span className="mb-1 block text-xs font-medium text-text-muted">Salience</span>
            <input
              aria-label="Memory salience"
              type="number"
              min="0"
              step="0.1"
              value={editSalience}
              disabled={IS_DEMO}
              onChange={(event) => setEditSalience(event.target.value)}
              className="h-9 w-full rounded-sm border border-border bg-surface px-2 text-sm text-text-body focus-ring"
            />
          </label>
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" size="sm" onClick={() => setEditTarget(null)} disabled={editBusy}>
              Cancel
            </Button>
            <Button type="submit" size="sm" disabled={IS_DEMO || editBusy || !editContent.trim()}>
              Save memory
            </Button>
          </div>
        </form>
      </Dialog>

      <Dialog
        open={Boolean(deleteTarget)}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
        title="Delete memory"
        description="This permanently removes the selected memory."
      >
        <p className="mb-4 text-sm text-text-body">{deleteTarget?.content}</p>
        <div className="flex justify-end gap-2">
          <Button type="button" variant="ghost" size="sm" onClick={() => setDeleteTarget(null)} disabled={deleteBusy}>
            Cancel
          </Button>
          <Button type="button" variant="danger" size="sm" onClick={() => void confirmDelete()} disabled={IS_DEMO || deleteBusy}>
            Delete
          </Button>
        </div>
      </Dialog>
    </div>
  );
}

import { Boxes, CheckCircle2, Pencil, Plug, Trash2, XCircle } from "lucide-react";
import type { LLMProviderProfile, LLMTestResult } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { IS_DEMO } from "@/lib/runtime";

type TestState = LLMTestResult | { error: string };

interface ProviderTableProps {
  providers: LLMProviderProfile[];
  tests: Record<string, TestState>;
  onToggle: (provider: LLMProviderProfile, enabled: boolean) => void;
  onTest: (id: string) => void;
  onManageModels: (provider: LLMProviderProfile) => void;
  onEdit: (provider: LLMProviderProfile) => void;
  onDelete: (provider: LLMProviderProfile) => void;
}

function testLabel(test: TestState): string {
  if ("error" in test) return test.error;
  const latency = test.latency_ms ? ` · ${test.latency_ms}ms` : "";
  const host = test.endpoint_host ? ` · ${test.endpoint_host}` : "";
  const missing =
    "missing_required" in test && test.missing_required?.length
      ? ` · 缺少 ${test.missing_required.join(", ")}`
      : "";
  const detail =
    "error_message" in test && test.error_message?.trim() ? ` · ${test.error_message.trim()}` : "";
  // Provider-level probe reports a model catalog count; model-level chat probe reports a model id.
  const model = "model" in test && test.model ? ` · model=${test.model}` : "";
  const models =
    "models_count" in test && typeof test.models_count === "number"
      ? ` · models=${test.models_count}`
      : "";
  return `${test.ok ? "✓" : "✗"} ${test.status}${missing}${model}${models}${detail}${latency}${host}`;
}

/** SRB-style provider table: one row per provider with inline row actions. */
export function ProviderTable({
  providers,
  tests,
  onToggle,
  onTest,
  onManageModels,
  onEdit,
  onDelete,
}: ProviderTableProps) {
  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <Table>
        <THead>
          <TR>
            <TH>Provider</TH>
            <TH>Endpoint</TH>
            <TH>Key</TH>
            <TH className="text-center">Models</TH>
            <TH className="text-center">Enabled</TH>
            <TH className="text-right">Actions</TH>
          </TR>
        </THead>
        <TBody>
          {providers.map((p) => {
            const test = tests[p.id];
            return (
              <TR key={p.id}>
                <TD>
                  <div className="font-medium text-text">{p.label || p.id}</div>
                  <div className="text-xs text-text-faint">
                    {p.id} · {p.type}
                  </div>
                  {test && (
                    <div className="mt-1 text-xs text-text-muted" role="status">
                      {testLabel(test)}
                    </div>
                  )}
                </TD>
                <TD className="max-w-[14rem] truncate text-xs text-text-muted">{p.base_url || "—"}</TD>
                <TD>
                  {p.has_key ? (
                    <Badge variant="success">
                      <CheckCircle2 className="h-3.5 w-3.5" /> set
                    </Badge>
                  ) : (
                    <Badge>
                      <XCircle className="h-3.5 w-3.5" /> none
                    </Badge>
                  )}
                </TD>
                <TD className="text-center text-text-muted">
                  {p.models.length}
                  {p.models.length === 0 && p.type !== "ollama" && (
                    <div className="mt-0.5 text-[10px] text-danger">需添加模型</div>
                  )}
                </TD>
                <TD className="text-center">
                  <Switch
                    checked={p.enabled}
                    disabled={IS_DEMO}
                    onCheckedChange={(v) => onToggle(p, v)}
                    aria-label={`Toggle ${p.id}`}
                  />
                </TD>
                <TD>
                  <div className="flex items-center justify-end gap-1">
                    <Button variant="ghost" size="icon" type="button" aria-label={`Test ${p.id}`} onClick={() => onTest(p.id)} disabled={IS_DEMO}>
                      <Plug className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      type="button"
                      aria-label={`Manage models ${p.id}`}
                      title="Configure models"
                      onClick={() => onManageModels(p)}
                    >
                      <Boxes className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="icon" type="button" aria-label={`Edit ${p.id}`} onClick={() => onEdit(p)} disabled={IS_DEMO}>
                      <Pencil className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="icon" type="button" aria-label={`Delete ${p.id}`} onClick={() => onDelete(p)} disabled={IS_DEMO}>
                      <Trash2 className="h-4 w-4 text-danger" />
                    </Button>
                  </div>
                </TD>
              </TR>
            );
          })}
        </TBody>
      </Table>
    </div>
  );
}

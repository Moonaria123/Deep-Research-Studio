import { Eye } from "lucide-react";
import { IS_DEMO } from "@/lib/runtime";

export function DemoNotice() {
  if (!IS_DEMO) return null;
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-2 gap-y-1 border-b border-warning/40 bg-warning/10 px-4 py-2 text-center text-xs text-text-body" role="note" aria-label="Demo preview notice">
      <Eye className="h-4 w-4 text-warning" aria-hidden />
      <strong>UI preview only / 仅供界面预览</strong>
      <span>Preset data · No model calls · No keys · Nothing is saved / 预设数据 · 不调用模型 · 无密钥 · 不保存数据</span>
    </div>
  );
}

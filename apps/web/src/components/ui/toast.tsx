import { useEffect, useState } from "react";
import { CheckCircle2, X, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export type ToastKind = "success" | "error" | "info";

export interface ToastItem {
  id: number;
  kind: ToastKind;
  message: string;
}

type Listener = (toasts: ToastItem[]) => void;

// Module-level toast store. Keeps the public API (toast.success/error) tiny and
// dependency-free while a single <Toaster /> subscribes for rendering.
let store: ToastItem[] = [];
const listeners = new Set<Listener>();
let nextId = 1;

function emit(): void {
  const snapshot = [...store];
  listeners.forEach((listener) => listener(snapshot));
}

function dismiss(id: number): void {
  store = store.filter((item) => item.id !== id);
  emit();
}

function push(kind: ToastKind, message: string): number {
  const id = nextId++;
  store = [...store, { id, kind, message }];
  emit();
  setTimeout(() => dismiss(id), 4000);
  return id;
}

export const toast = {
  success: (message: string) => push("success", message),
  error: (message: string) => push("error", message),
  info: (message: string) => push("info", message),
  dismiss,
};

/** Global toast surface. Mount once near the app root. */
export function Toaster() {
  const [items, setItems] = useState<ToastItem[]>([]);
  useEffect(() => {
    listeners.add(setItems);
    setItems([...store]);
    return () => {
      listeners.delete(setItems);
    };
  }, []);

  if (items.length === 0) return null;
  return (
    <div
      className="fixed bottom-4 right-4 z-[60] flex w-80 flex-col gap-2"
      role="region"
      aria-label="Notifications"
    >
      {items.map((item) => (
        <div
          key={item.id}
          role="status"
          className={cn(
            "flex items-start gap-2 rounded-md border bg-surface px-3 py-2 text-sm shadow-card",
            item.kind === "error" ? "border-danger" : "border-border",
          )}
        >
          {item.kind === "error" ? (
            <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-danger" />
          ) : (
            <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" />
          )}
          <span className="flex-1 text-text-body">{item.message}</span>
          <button
            type="button"
            aria-label="Dismiss notification"
            onClick={() => dismiss(item.id)}
            className="text-text-faint hover:text-text"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      ))}
    </div>
  );
}

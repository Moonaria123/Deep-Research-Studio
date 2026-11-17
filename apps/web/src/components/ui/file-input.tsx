import * as React from "react";
import { cn } from "@/lib/utils";

type FileInputProps = Omit<React.InputHTMLAttributes<HTMLInputElement>, "type" | "value" | "defaultValue" | "multiple"> & {
  fileName?: string;
};

export const FileInput = React.forwardRef<HTMLInputElement, FileInputProps>(
  ({ className, fileName, disabled, ...props }, ref) => (
    <span
      className={cn(
        "relative flex h-9 min-w-0 w-full items-center gap-3 rounded-sm border border-border bg-surface px-3 text-sm text-text-body focus-within:ring-2 focus-within:ring-[color:var(--ring)] focus-within:ring-offset-2 focus-within:ring-offset-[color:var(--bg)]",
        disabled && "opacity-50",
        className,
      )}
    >
      <span aria-hidden="true" className="shrink-0 font-medium">Choose file</span>
      <span aria-hidden="true" className="min-w-0 truncate text-text-muted">{fileName || "No file chosen"}</span>
      {/* Keep native file selection and keyboard access while controlling the visible UI language. */}
      <input
        {...props}
        ref={ref}
        type="file"
        disabled={disabled}
        title={fileName || "No file chosen"}
        className="absolute inset-0 h-full w-full cursor-pointer opacity-0 disabled:cursor-not-allowed"
      />
    </span>
  ),
);
FileInput.displayName = "FileInput";

import { Layers, Telescope, Users, Zap, type LucideIcon } from "lucide-react";
import type { Mode } from "@/lib/types";

export const MODE_SHORT_LABEL: Record<Mode, string> = {
  fast: "Fast",
  multiple: "Multiple",
  deep: "Deep",
  council: "Council",
};

export const MODE_ARIA_LABEL: Record<Mode, string> = {
  fast: "Fast Search",
  multiple: "Multiple Search",
  deep: "Deep Research",
  council: "Models Council",
};

export const MODE_ICON: Record<Mode, LucideIcon> = {
  fast: Zap,
  multiple: Layers,
  deep: Telescope,
  council: Users,
};

const MODES: Mode[] = ["fast", "multiple", "deep", "council"];

/** Narrow an API string to a known research mode. */
export function isMode(value: string | null | undefined): value is Mode {
  return typeof value === "string" && (MODES as string[]).includes(value);
}

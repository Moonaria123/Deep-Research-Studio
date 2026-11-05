import { NavLink } from "react-router-dom";
import {
  Compass,
  History,
  Library,
  Brain,
  Settings,
  PanelLeftClose,
  PanelLeftOpen,
  Network,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useSidebarCollapsed } from "@/hooks/useSidebarCollapsed";
import { ThemeToggle } from "./ThemeToggle";

const NAV = [
  { to: "/", label: "New", icon: Compass, end: true },
  { to: "/history", label: "History", icon: History },
  { to: "/library", label: "Library", icon: Library },
  { to: "/memory", label: "Memory", icon: Brain },
  { to: "/graph", label: "Knowledge Graph", icon: Network },
  { to: "/settings", label: "Settings", icon: Settings },
];

interface SidebarProps {
  variant?: "docked" | "drawer";
  onClose?: () => void;
  onNavigate?: () => void;
}

export function Sidebar({ variant = "docked", onClose, onNavigate }: SidebarProps) {
  if (variant === "drawer") {
    return (
      <SidebarContent
        collapsed={false}
        variant="drawer"
        onClose={onClose}
        onNavigate={onNavigate}
      />
    );
  }
  return <DockedSidebar />;
}

function DockedSidebar() {
  const { collapsed, toggle } = useSidebarCollapsed();
  return <SidebarContent collapsed={collapsed} variant="docked" onToggleCollapse={toggle} />;
}

function SidebarContent({
  collapsed,
  variant,
  onToggleCollapse,
  onClose,
  onNavigate,
}: {
  collapsed: boolean;
  variant: "docked" | "drawer";
  onToggleCollapse?: () => void;
  onClose?: () => void;
  onNavigate?: () => void;
}) {
  const drawer = variant === "drawer";
  return (
    <nav
      aria-label="Primary"
      aria-expanded={!collapsed}
      className={cn(
        "flex h-full min-w-0 shrink-0 flex-col border-r border-border bg-bg-sidebar",
        drawer ? "w-full" : "transition-[width] duration-200",
        !drawer && (collapsed ? "w-16" : "w-[240px]"),
      )}
    >
      <div
        className={cn(
          "flex items-center py-4",
          collapsed ? "justify-center px-2" : "gap-2 px-5",
          drawer && "justify-between px-4",
        )}
      >
        <div className={cn("flex min-w-0 items-center", collapsed ? "justify-center" : "gap-2")}>
          <div
            className="grid h-8 w-8 shrink-0 place-items-center rounded-md bg-accent text-accent-contrast font-bold"
            title="Deep Research Studio"
          >
            R
          </div>
          {!collapsed && <span className="truncate text-sm font-semibold text-text">Deep Research Studio</span>}
        </div>
        {drawer && (
          <button
            type="button"
            aria-label="Close navigation"
            onClick={onClose}
            className="grid h-8 w-8 shrink-0 place-items-center rounded-md text-text-muted hover:bg-bg-subtle focus-ring"
          >
            <X className="h-[18px] w-[18px]" aria-hidden />
          </button>
        )}
      </div>
      <ul className={cn("flex-1 space-y-1 py-2", collapsed ? "px-2" : "px-3")}>
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={end}
              onClick={onNavigate}
              title={collapsed ? label : undefined}
              aria-label={collapsed ? label : undefined}
              className={({ isActive }) =>
                cn(
                  "flex items-center rounded-md text-sm font-medium transition-colors focus-ring",
                  collapsed ? "justify-center px-2 py-2" : "gap-3 px-3 py-2",
                  isActive
                    ? "bg-accent-soft text-accent"
                    : "text-text-muted hover:bg-bg-subtle hover:text-text-body",
                )
              }
            >
              <Icon className="h-[18px] w-[18px] shrink-0" aria-hidden />
              {!collapsed && <span>{label}</span>}
            </NavLink>
          </li>
        ))}
      </ul>
      <div
        className={cn(
          "border-t border-border py-3",
          collapsed ? "flex flex-col items-center gap-2 px-2" : "flex items-center justify-between px-3",
        )}
      >
        <ThemeToggle compact={collapsed} />
        {!drawer && (
          <button
            className="grid h-8 w-8 place-items-center rounded-md text-text-muted hover:bg-bg-subtle focus-ring"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            type="button"
            onClick={onToggleCollapse}
          >
            {collapsed ? (
              <PanelLeftOpen className="h-[18px] w-[18px]" aria-hidden />
            ) : (
              <PanelLeftClose className="h-[18px] w-[18px]" aria-hidden />
            )}
          </button>
        )}
      </div>
    </nav>
  );
}

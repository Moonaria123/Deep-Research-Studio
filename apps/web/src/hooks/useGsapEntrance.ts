import { useEffect, useRef } from "react";
import { gsap } from "gsap";

/**
 * Fade + 8px rise entrance for direct children with stagger, per design-system motion.
 * Respects prefers-reduced-motion (skips animation entirely).
 */
export function useGsapEntrance<T extends HTMLElement>(deps: unknown[] = []) {
  const ref = useRef<T>(null);
  useEffect(() => {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const el = ref.current;
    if (!el) return;
    const targets = el.querySelectorAll("[data-anim]");
    if (!targets.length) return;
    const ctx = gsap.context(() => {
      gsap.from(targets, {
        opacity: 0,
        y: 8,
        duration: 0.24,
        ease: "power2.out",
        stagger: 0.06,
      });
    }, el);
    return () => ctx.revert();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return ref;
}

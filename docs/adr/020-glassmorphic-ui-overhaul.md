# ADR-020: Glassmorphic UI/UX Overhaul

**Date:** 2026-08-08
**Status:** Accepted
**Deciders:** Jatin Gupta

## Context

The Next.js dashboard (ADR-017) was functional but visually unpolished — basic bordered cards, no animations, inconsistent spacing, and a utilitarian look that didn't match the sophistication of the backend pipeline. For a flagship portfolio project, the UI needs to feel premium and showcase modern frontend engineering.

## Decision

Rewrite all dashboard pages with a glassmorphic design system and framer-motion animations:

### Design System
- **Glassmorphic cards**: `backdrop-blur-md` + semi-transparent gradient backgrounds (`from-zinc-900/60 via-zinc-900/40 to-zinc-800/30`) with subtle `ring-1 ring-zinc-700/30` borders
- **Hover states**: scale + blue glow shadows (`shadow-[0_0_20px_rgba(59,130,246,0.08)]`)
- **Color accents**: blue-500 primary, emerald-400 success, amber-400 warning, red-400 danger — used sparingly in icon backgrounds with `ring-1` styling
- **Typography**: 11px uppercase tracking-wider labels, 3xl bold tracking-tight values, mono for data

### Animations (framer-motion)
- `fadeUp` stagger variants with per-element delay
- `layoutId` for animated tab indicators and filter pill backgrounds
- `AnimatePresence` for page/content transitions
- Animated SVG gauge rings (`strokeDashoffset` transition) for spoof scores
- `motion.div` width animations for progress bars
- Shimmer CSS animation for processing states

### Pages Rewritten
1. **globals.css** — full CSS redesign with `.glass`, `.glass-hover`, glow, shimmer, gradient-border, grid background, pulse-dot classes
2. **sidebar.tsx** — active nav indicator bar, pulse-dot API status, backdrop-blur mobile overlay
3. **page.tsx (home)** — stat cards, pipeline architecture visualization, quick actions, API status
4. **upload-zone.tsx** — animated drag-over with blue glow, AnimatePresence toasts
5. **result-detail.tsx** — modal animations, layoutId tab indicator, SVG GaugeCard, staggered field cards
6. **job-card.tsx** — fade+slide animation, shimmer progress bar
7. **cases/page.tsx** — animated filter pills, glassmorphic table, staggered row animations
8. **forensics/page.tsx** — SVG spoof gauge, risk distribution cards with icons, animated evidence items, color-coded component scores
9. **metrics/page.tsx** — 6 stat cards, animated bar charts for doc types/risk/decisions
10. **settings/page.tsx** — sectioned cards with icon headers, animated status indicator, save confirmation

## Consequences

### Positive
- Premium visual quality befitting a portfolio showcase
- Consistent design language across all pages
- Smooth 60fps animations via framer-motion (GPU-accelerated transforms)
- Data visualization (gauges, bars) communicates pipeline results at a glance

### Negative
- framer-motion adds ~30KB to the client bundle
- backdrop-blur can be expensive on low-end GPUs (mitigated by using it sparingly)
- Animations may be distracting for users who prefer reduced motion (could add `prefers-reduced-motion` media query in future)

## Dependencies Added
- `framer-motion` (animation library, already in package.json)

# ADR-017: Next.js 16 Dashboard for KYC Intelligence Platform

## Status
Accepted

## Date
2026-08-08

## Context
The platform needed a modern web dashboard for document upload, case tracking, forensics visualization, and pipeline metrics. The existing `apps/review_ui/` is a Streamlit-based HITL review interface for analysts — it lacks the polish and UX needed for a showcase/production dashboard.

Key requirements:
- Real-time job status tracking with auto-polling
- Drag-and-drop document upload
- Tabbed result inspection (fields, forensics, calibration, cross-doc, policy)
- Dark theme, responsive layout
- Decision timeline visualization per case
- Forensics risk distribution dashboard
- Pipeline metrics aggregation

## Decision
Build a standalone Next.js 16 dashboard in `dashboard/` using:

- **Next.js 16.3** (App Router) — latest framework with Turbopack, React 19
- **Tailwind CSS v4** — utility-first styling with `@tailwindcss/postcss`
- **Lucide React** — consistent icon set
- **clsx** — conditional class composition
- **localStorage** — client-side job persistence (no auth needed for local dev)
- **Fetch API** — direct calls to FastAPI backend at `http://localhost:8000`

Architecture:
- 6 routes: `/` (dashboard), `/upload`, `/cases`, `/forensics`, `/metrics`, `/settings`
- Shared sidebar with nav links and API status indicator
- `useJobs` hook manages upload → poll → complete lifecycle
- `ResultDetail` modal with 5 tabbed panels for deep inspection
- `DecisionTimeline` component showing each pipeline stage result
- All state in `localStorage` keyed as `kyc_jobs` — no database needed

## Alternatives Considered

1. **Streamlit** — Already used for review UI. Poor for complex dashboards, no component reuse, limited layout control.
2. **Remix** — Strong full-stack framework but overkill for a client-side dashboard consuming an existing API.
3. **Vite + React** — Lighter but lacks Next.js file-based routing, SSR capability for future deployment.

## Consequences

### Positive
- Modern, portfolio-worthy UI with dark theme
- Component-based — each page is self-contained, easy to extend
- Hot reload via Turbopack makes iteration fast
- Can deploy to Vercel/Netlify as static export
- TypeScript throughout — full type safety with API types

### Negative
- Adds Node.js dependency alongside Python stack
- localStorage has ~5MB limit — adequate for job metadata but won't scale to thousands of cases
- No SSR benefit currently (all pages are `"use client"`)

### Risks
- API types (`types/api.ts`) must stay in sync with FastAPI response schemas manually

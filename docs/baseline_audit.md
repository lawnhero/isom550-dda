# Baseline Audit Freeze

This document defines the baseline metrics captured before the revamp.

## Baseline metrics
- Total historic query events
- Average query length (words)
- Query length distribution (`short`, `medium`, `long`)
- Timestamp range of historical usage

## Script
- Use `scripts/baseline_metrics.py` to generate `analytics/baseline_metrics.json`.

## Why this matters
These baseline numbers provide a fixed reference point for evaluating whether:
- trust and source transparency increase engagement,
- hint-first tutoring improves continuation rate,
- adaptive tutoring reduces unresolved interactions.

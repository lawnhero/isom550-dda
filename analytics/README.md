# Analytics Outputs

This folder stores generated analytics artifacts:

- `baseline_metrics.json`: snapshot from pre-revamp query logs
- `weekly_report.json`: rolling 7-day summary of student usage and friction

Generate files with:

```bash
python scripts/baseline_metrics.py --mongo-uri "<YOUR_MONGO_URI>"
python scripts/generate_weekly_report.py --mongo-uri "<YOUR_MONGO_URI>"
```

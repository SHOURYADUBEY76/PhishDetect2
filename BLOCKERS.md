# Project Issues & Fallback Audit Log (BLOCKERS.md)

This log records any external feed download failures, dataset missing conditions, and the corresponding fallback strategies implemented.

| Date | Item | Issue Encountered | Resolution / Fallback Implemented | Status |
| :--- | :--- | :--- | :--- | :--- |
| Initial Setup | Remote Feed Access | Network latency/rate-limiting potential on live feeds | Built-in offline seed generators and local cache in `data/raw/` with automatic fallback verification | Resolved |

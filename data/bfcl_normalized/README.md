# Normalized BFCL Input Format

Each line in `sample.jsonl` is one JSON object with this schema:

- `id`: stable example identifier
- `user_request`: current user utterance
- `history`: prior dialogue turns as `{role, content}`
- `tools`: tool list with `name`, `description`, and JSON-schema-like `parameters`
- `metadata`: bookkeeping fields, including an optional synthetic `failure_signal`
- `gold`: optional analysis fields for manual inspection or lightweight debugging

This format is intentionally simpler than the official BFCL internal data layout. The point is to let you iterate on the escalation policy locally before wiring it back into the full BFCL pipeline.

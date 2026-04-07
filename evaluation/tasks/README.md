# Tasks

Task adapters normalize different datasets into a common page schema:

- page image or rendered PDF
- target Markdown
- target LaTeX spans
- optional region annotations
- latency budget and difficulty tags

Task adapters should not redefine benchmark policy. They should consume protocol versions from `benchmark/protocols/`.

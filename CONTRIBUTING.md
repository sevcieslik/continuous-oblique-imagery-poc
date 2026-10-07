# Contributing

This is an internal experimental repository. Keep changes small and evidence-driven.

For behavioural changes:

1. state which product rule is being changed;
2. add/retain QA fields rather than hiding uncertainty;
3. run a smoke test on a small subset;
4. compare against the frozen benchmark/adjudication set where available;
5. do not use manual production as unquestioned truth;
6. do not commit client raw data.

Versioned POC files are retained for history, but new development should prefer a branch from the current
implementation and eventually move toward a package/module structure rather than another monolithic copy.

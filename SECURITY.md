# Security and data handling

This repository is intended to be private/internal.

Do not commit:
- raw client imagery;
- LAS/LAZ production data;
- EO/staking files containing client/project-sensitive information;
- model credentials, Hugging Face tokens, GitHub tokens, or passwords;
- full production output folders.

The `samples/` directory contains anonymised/synthetic examples only. `.gitignore` blocks the
main production data folders and common large geospatial/image formats by default.

If a real QA image is needed for an issue, crop or redact it first and confirm that the project
permits it to be stored in GitHub.

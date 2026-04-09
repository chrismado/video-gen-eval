# Security Policy

## Reporting a Vulnerability

Please report suspected vulnerabilities by opening a private security advisory on GitHub or by contacting the maintainer directly before public disclosure. Include reproduction steps, affected files, and any proof-of-concept details needed to validate the issue.

## Audit Summary

- No hardcoded API keys, tokens, or cloud credentials were found during the April 2026 hardening pass.
- No uses of `eval()`, unsafe `yaml.load`, shell injection with user-controlled `shell=True`, or insecure temporary-file patterns were found in the repository code.
- The pipeline writes JSON reports and temporary image files, but does not deserialize untrusted pickles or execute external commands from user input.
- Dependency auditing should be rerun regularly with `pip-audit -r requirements.txt` and `bandit -r . -ll`.

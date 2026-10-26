# Contributing to HealthScan

Thank you for helping improve HealthScan. Contributions are welcome, but
changes must preserve the project's safety notice and avoid presenting
experimental measurements as medical diagnoses.

## Before you start

- Check existing issues and pull requests to avoid duplicated work.
- For substantial changes, open an issue first to agree on the approach.
- Never include personal health data, camera recordings, credentials, or other
  private information in an issue or pull request.
- Review the [Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md).

## Development workflow

1. Fork the repository and create a branch from the default branch:
   `git switch -c feature/short-description`.
2. Create a local virtual environment and install `requirements.txt`.
3. Keep changes focused and follow the existing Python style.
4. Run `python main.py` when camera behavior is affected and record the
   environment and validation steps in the pull request.
5. Update the README or other documentation when behavior or setup changes.
6. Submit a pull request using the repository template.

## Pull requests

A pull request should include:

- A concise description of the problem and solution.
- The relevant issue number, if one exists.
- Tests or manual checks performed.
- Known limitations and hardware or lighting conditions.
- Screenshots or recordings for visible changes, with no identifiable people
  unless they have given permission.

Maintainers may request changes, additional validation, or documentation before
merging. Pull requests are normally merged only after review and passing
checks.

## Reporting bugs and proposing features

Use the issue templates for reproducible bugs and focused feature proposals.
Include steps to reproduce, expected behavior, actual behavior, and relevant
environment details. Feature requests should explain the user problem rather
than prescribe an implementation.

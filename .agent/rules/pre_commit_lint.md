# Pre-Commit Lint & Format Gate

> [!CAUTION]
> These checks are **mandatory** before every `git commit`. Never commit code that fails any of these steps.

## Required Steps (in order)

Before creating a git commit, **always** run the following commands from the project root using the virtual environment:

1. **Ruff lint check** — catch code quality issues:

   ```bash
   ./venv/bin/ruff check .
   ```

   - If errors are found, fix them before proceeding.

2. **Ruff format** — enforce consistent code style:

   ```bash
   ./venv/bin/ruff format .
   ```

3. **Flake8 lint check** — catch any remaining issues ruff may miss:

   ```bash
   source venv/bin/activate && flake8 autopilot/ tests/ workflows/ --max-line-length=120 --extend-ignore=E402 --count
   ```

   - If errors are found, fix them before proceeding.

4. **Verify clean** — confirm zero issues remain:
   ```bash
   ./venv/bin/ruff check . && ./venv/bin/ruff format --check .
   ```

   - Only proceed with the commit if this outputs `All checks passed!` and all files are already formatted.

## Rules

- **NEVER** skip these steps, even for "small" or "trivial" changes.
- **ALWAYS** fix all reported issues before committing — do not use `# noqa` or `--no-verify` unless explicitly approved by the user.
- If `ruff format` and `flake8` disagree on a style rule (e.g., E203 slice spacing), **ruff format takes precedence** since it auto-applies the fix.
- Run all commands from the project root (`/Users/camilopiedra/Development/Autopilot`).

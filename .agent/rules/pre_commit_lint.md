# Pre-Commit Lint & Format Gate

> [!CAUTION]
> These checks are **mandatory** before every `git commit`. Never commit code that fails any of these steps.

## Required Steps (in order)

Before creating a git commit, **always** run the following commands from the project root using the virtual environment:

1. **Ruff format** — enforce consistent code style:

   ```bash
   ./venv/bin/ruff format .
   ```

2. **Ruff lint check with auto-fix** — catch and fix code quality issues:

   ```bash
   ./venv/bin/ruff check . --fix
   ```

   - If any remaining errors are reported after auto-fix, manually fix them before proceeding.

3. **Final verification** — confirm zero issues remain:

   ```bash
   ./venv/bin/ruff format --check . && ./venv/bin/ruff check .
   ```

   - Only proceed with the commit if both commands exit with code 0.

## Rules

- **NEVER** skip these steps, even for "small" or "trivial" changes.
- **ALWAYS** fix all reported issues before committing — do not use `# noqa` or `--no-verify` unless explicitly approved by the user.
- Run all commands from the project root (`/Users/camilopiedra/Development/Autopilot`).

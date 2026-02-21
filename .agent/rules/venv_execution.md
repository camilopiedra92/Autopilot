# Environment Execution Rules

> [!IMPORTANT]
> These rules enforce strict isolation and dependency management across the platform.

- **CRITICAL**: **ALWAYS** execute Python commands, tests (`pytest`), installations, and scripts within the activated virtual environment (`venv`).
- When running commands via the terminal loop or scripts, use the prefix `source venv/bin/activate && ` to guarantee that dependencies are correctly resolved.
- **NEVER** run `pip install` or `python` commands against the global system Python.
- If a missing dependency error occurs (e.g., `ModuleNotFoundError`), verify that you are running within the `venv` before attempting any fixes.

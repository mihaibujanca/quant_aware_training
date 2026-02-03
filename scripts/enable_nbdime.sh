#!/usr/bin/env bash
# Enable nbdime git integration for notebook diffs/merges.
# Safe to re-run.
set -euo pipefail

echo "Enabling nbdime git integration..."
nbdime config-git --enable

echo ""
echo "nbdime is now configured. Verify with:"
echo "  git config diff.jupyternotebook.command"
echo "  git config merge.jupyternotebook.command"

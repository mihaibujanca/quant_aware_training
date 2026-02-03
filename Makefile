.PHONY: brief repomap pair tooling

brief:
	scripts/make_session_brief.sh

repomap:
	scripts/make_repo_map.sh

pair:
	scripts/pair_notebooks.sh

tooling:
	@echo "Install dev dependencies:"
	@echo "  uv sync --group dev"
	@echo ""
	@echo "Setup pre-commit:"
	@echo "  pre-commit install"
	@echo ""
	@echo "Enable nbdime:"
	@echo "  scripts/enable_nbdime.sh"
	@echo ""
	@echo "Pair notebooks:"
	@echo "  scripts/pair_notebooks.sh"

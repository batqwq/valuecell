format:
	ruff format --config ./python/pyproject.toml ./python/ && uv run --directory ./python isort .

lint:
	ruff check --config ./python/pyproject.toml ./python/

test:
	uv run pytest ./python

searchXagent:
	cd python && uv run --env-file ../.env -m valuecell.agents.research_agent.search_x_agent

tradingagents:
	cd python/third_party/TradingAgents && uv run --env-file ../../.env -m adapter

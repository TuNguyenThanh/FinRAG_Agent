API ?= http://localhost:8000
KEY ?= dev-key

.PHONY: up down logs test lint ingest ask eval eval-answer

up:            ## build and start postgres, redis, api
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

ingest:        ## index the sample bank documents
	@for f in data/samples/*; do \
		curl -s -X POST $(API)/v1/documents -H "X-API-Key: $(KEY)" -F "file=@$$f"; echo; \
	done

ask:           ## make ask Q="Phí thường niên thẻ Gold?"
	curl -N -s -X POST $(API)/v1/chat -H "X-API-Key: $(KEY)" -H "Content-Type: application/json" \
		-d "{\"message\": \"$(Q)\"}"

eval:          ## retrieval metrics (no LLM cost)
	uv run python evals/run_eval.py --mode retrieval --base-url $(API) --api-key $(KEY)

eval-answer:   ## end-to-end answer metrics (uses LLM tokens)
	uv run python evals/run_eval.py --mode answer --judge --base-url $(API) --api-key $(KEY)

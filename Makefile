# setup-work — основной пользовательский интерфейс.
#
# Каждый таргет — тонкая обёртка над Python CLI (`setup-work`).
# Параметры (MODEL, ENGINE, TASK, ...) передаются в CLI через переменные
# окружения SETUP_WORK_* и никогда не подставляются в shell-команды.
#
# Полный список таргетов и примеры: `make help`.

SHELL := /bin/sh
.DEFAULT_GOAL := help

UV ?= $(shell command -v uv 2>/dev/null || echo "$(HOME)/.local/bin/uv")
CLI := $(UV) run --no-sync --quiet setup-work
VENV_STAMP := .venv/.setup-work-synced

# ---- параметры -> окружение (без подстановки в shell) ------------------------
export SETUP_WORK_MODEL          = $(MODEL)
export SETUP_WORK_ENGINE         = $(ENGINE)
export SETUP_WORK_TASK           = $(TASK)
export SETUP_WORK_CONTEXT        = $(CONTEXT)
export SETUP_WORK_CONCURRENCY    = $(CONCURRENCY)
export SETUP_WORK_TP             = $(TP)
export SETUP_WORK_KV_CACHE_DTYPE = $(KV_CACHE_DTYPE)
export SETUP_WORK_LIMIT          = $(LIMIT)
export SETUP_WORK_EXPLAIN        = $(EXPLAIN)
export SETUP_WORK_NAME           = $(NAME)
export SETUP_WORK_VERSION        = $(VERSION)
export SETUP_WORK_REF            = $(REF)
export SETUP_WORK_ALLOW_BUILD    = $(ALLOW_BUILD)
export SETUP_WORK_INSTANCE       = $(INSTANCE)
export SETUP_WORK_HOST           = $(HOST)
export SETUP_WORK_PORT           = $(PORT)
export SETUP_WORK_GPU_UTIL       = $(GPU_UTIL)
export SETUP_WORK_GPU_LAYERS     = $(GPU_LAYERS)
export SETUP_WORK_THREADS        = $(THREADS)
export SETUP_WORK_SERVED_NAME    = $(SERVED_NAME)
export SETUP_WORK_FORCE          = $(FORCE)
export SETUP_WORK_PLAN           = $(PLAN)
export SETUP_WORK_URL            = $(URL)
export SETUP_WORK_LINES          = $(LINES)
export SETUP_WORK_FOLLOW         = $(FOLLOW)
export SETUP_WORK_DRY_RUN        = $(DRY_RUN)
export SETUP_WORK_YES            = $(YES)
export SETUP_WORK_JSON           = $(JSON)
export SETUP_WORK_VERBOSE        = $(VERBOSE)
export SETUP_WORK_ENGINE_ARGS    = $(ENGINE_ARGS)
ifeq ($(FOREGROUND),1)
export SETUP_WORK_DETACH = 0
endif
ifneq ($(HARDWARE),)
export SETUP_WORK_HARDWARE_FILE = $(abspath $(HARDWARE))
endif

define require
	@if [ -z "$$SETUP_WORK_$(2)" ]; then \
		echo "error: $(1) is required, e.g. make $@ $(1)=$(3)" >&2; exit 2; fi
endef

.PHONY: help init registries bootstrap sync doctor hardware hardware-json version \
	config-show config-validate models recommend model-pull model-list model-info \
	model-remove zip-join engine-install engine-status serve serve-auto serve-stop \
	serve-status serve-logs serve-test harness-list harness-install harness-update \
	harness-remove harness-doctor test lint format check clean

CONFIG_DIR = $${SETUP_WORK_USER_CONFIG_DIR:-$${XDG_CONFIG_HOME:-$$HOME/.config}/setup-work}

##@ 1. Настройка (все настройки — в ~/.config/setup-work/)
help: ## Показать эту справку
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
		/^[a-zA-Z0-9_-]+:.*## / { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf '\n\033[1mС чего начать\033[0m\n'
	@printf '  make init          # создать ~/.config/setup-work/ и заполнить\n'
	@printf '  make bootstrap     # pip/npm/uv + uv + окружение + Node.js\n'
	@printf '  make doctor\n'
	@printf '\n  DRY_RUN=1 — только показать действия · YES=1 — подтвердить всё · JSON=1 — JSON\n\n'

init: ## Создать ~/.config/setup-work/{bootstrap.env,corporate.yaml,harnesses.yaml} из шаблонов
	@d="$(CONFIG_DIR)"; mkdir -p "$$d"; \
	for f in bootstrap.env corporate.yaml harnesses.yaml; do \
		if [ -e "$$d/$$f" ]; then echo "  keep    $$d/$$f (already exists)"; \
		else cp "examples/corp/$$f" "$$d/$$f" && echo "  created $$d/$$f"; fi; \
	done; \
	echo "Fill in the CHANGE-ME / example values, then: make bootstrap"

registries: ## Записать PyPI/npm из bootstrap.env в pip.conf, uv.toml и .npmrc
	@sh scripts/bootstrap.sh --registries-only

bootstrap: ## registries → uv → окружение проекта → Node.js (из corporate.yaml)
	@sh scripts/bootstrap.sh
	@$(CLI) bootstrap

sync: $(VENV_STAMP) ## Пересобрать окружение проекта (uv sync через корпоративный PyPI)

# uv.lock records pypi.org; bootstrap.sh re-resolves it against the index in uv.toml.
$(VENV_STAMP): uv.lock pyproject.toml
	@sh scripts/bootstrap.sh --sync-only

doctor: $(VENV_STAMP) ## Диагностика: инструменты, pip/uv/npm, HDFS, железо, движки
	@$(CLI) doctor

##@ 2. Модели и движки (из corporate.yaml)
models: $(VENV_STAMP) ## Каталог моделей [TASK=]
	@$(CLI) models

recommend: $(VENV_STAMP) ## Подобрать модель под железо [TASK= CONTEXT= ENGINE= CONCURRENCY= TP= EXPLAIN=1 HARDWARE=]
	@$(CLI) recommend

model-pull: $(VENV_STAMP) ## Скачать и распаковать веса: MODEL=id:variant
	$(call require,MODEL,MODEL,qwen3-8b:q4_k_m)
	@$(CLI) model pull

model-list: $(VENV_STAMP) ## Распакованные модели
	@$(CLI) model list

model-info: $(VENV_STAMP) ## Метаданные модели: MODEL=id[:variant]
	$(call require,MODEL,MODEL,qwen3-8b)
	@$(CLI) model info

model-remove: $(VENV_STAMP) ## Удалить распакованную модель: MODEL=id:variant
	$(call require,MODEL,MODEL,qwen3-8b:q4_k_m)
	@$(CLI) model remove

zip-join: ## Склеить части архива (.zip, .z01, …) после подтверждения: ZIP=префикс|файл.zip|папка [OUT=]
	@if [ -z "$(ZIP)" ]; then \
		echo "error: ZIP is required, e.g. make $@ ZIP=/data/Qwen3.8-Flash" >&2; exit 2; fi
	@python3 scripts/join_zip.py "$(ZIP)" $(if $(OUT),-o "$(OUT)") $(if $(filter 1,$(ZIP_LAST)),--zip-last)

engine-install: $(VENV_STAMP) ## Установить движок: ENGINE=llama.cpp|vllm
	$(call require,ENGINE,ENGINE,llama.cpp)
	@$(CLI) engine install

engine-status: $(VENV_STAMP) ## Установленные движки
	@$(CLI) engine status

##@ 3. Сервер (127.0.0.1 по умолчанию)
serve: $(VENV_STAMP) ## Запустить: MODEL= [ENGINE= CONTEXT= PORT= TP= ENGINE_ARGS= FOREGROUND=1 PLAN=1]
	$(call require,MODEL,MODEL,qwen3-8b:q4_k_m)
	@$(CLI) serve

serve-auto: $(VENV_STAMP) ## Запустить лучшую скачанную модель для TASK=
	@$(CLI) server auto

serve-stop: $(VENV_STAMP) ## Остановить [INSTANCE=default]
	@$(CLI) server stop

serve-status: $(VENV_STAMP) ## Статус
	@$(CLI) server status

serve-logs: $(VENV_STAMP) ## Логи [LINES=100 FOLLOW=1]
	@$(CLI) server logs

serve-test: $(VENV_STAMP) ## Тестовый запрос к OpenAI-совместимому API [URL=]
	@$(CLI) server test

##@ 4. Харнессы и skillhub (рецепты — в ~/.config/setup-work/harnesses.yaml)
harness-list: $(VENV_STAMP) ## Рецепты и что установлено
	@$(CLI) harness list

harness-install: $(VENV_STAMP) ## Установить: NAME= [VERSION= REF=]
	$(call require,NAME,NAME,qwen-code)
	@$(CLI) harness install

harness-update: $(VENV_STAMP) ## Обновить: NAME= [VERSION= REF=]
	$(call require,NAME,NAME,qwen-code)
	@$(CLI) harness update

harness-remove: $(VENV_STAMP) ## Удалить: NAME=
	$(call require,NAME,NAME,qwen-code)
	@$(CLI) harness remove

harness-doctor: $(VENV_STAMP) ## Проверить (все установленные, если NAME пуст)
	@$(CLI) harness doctor

##@ Прочее
hardware: $(VENV_STAMP) ## Железо [JSON=1]
	@$(CLI) hardware

hardware-json: $(VENV_STAMP) ## Железо в JSON (для HARDWARE= на другой машине)
	@$(CLI) hardware --json

config-show: $(VENV_STAMP) ## Итоговая конфигурация и откуда она взята
	@$(CLI) config show

config-validate: $(VENV_STAMP) ## Проверить YAML-конфигурацию
	@$(CLI) config validate

version: $(VENV_STAMP) ## Версия
	@$(CLI) version

test: $(VENV_STAMP) ## Тесты
	@$(UV) run --no-sync pytest

lint: $(VENV_STAMP) ## Линтер и проверка форматирования
	@$(UV) run --no-sync ruff check .
	@$(UV) run --no-sync ruff format --check .

format: $(VENV_STAMP) ## Отформатировать код
	@$(UV) run --no-sync ruff format .
	@$(UV) run --no-sync ruff check --fix .

check: lint test config-validate ## Линтер + тесты + конфигурация

clean: ## Удалить кэши сборки и установщика (модели и харнессы не трогает)
	@rm -rf .pytest_cache .ruff_cache build dist
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@if [ -f "$(VENV_STAMP)" ]; then $(CLI) clean; fi

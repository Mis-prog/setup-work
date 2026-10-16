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
CLI := $(UV) run --frozen --quiet setup-work
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
export SETUP_WORK_PROFILE        = $(PROFILE)
export SETUP_WORK_WITH_OPTIONAL  = $(WITH_OPTIONAL)
export SETUP_WORK_ALLOW_SUDO     = $(ALLOW_SUDO)
export SETUP_WORK_ALLOW_DOWNLOADS = $(ALLOW_DOWNLOADS)
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

.PHONY: help version doctor hardware hardware-json config-show config-validate \
	bootstrap bootstrap-python bootstrap-node bootstrap-tools engine-install engine-status \
	models recommend model-pull model-list model-info model-remove \
	harness-list harness-install harness-update harness-remove harness-doctor \
	serve serve-auto serve-stop serve-status serve-logs serve-test \
	setup profiles profile-show clean test lint format check sync \
	corp-setup corp-registries corp-status corp-env

##@ Общее
help: ## Показать эту справку
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
		/^[a-zA-Z0-9_-]+:.*## / { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf '\n\033[1mОбщие переменные\033[0m\n'
	@printf '  DRY_RUN=1  только показать действия     YES=1  подтвердить всё     JSON=1  вывод в JSON\n'
	@printf '  HARDWARE=examples/hardware/<f>.json   планировать под записанную машину, а не текущую\n'
	@printf '\n\033[1mПримеры\033[0m\n'
	@printf '  make doctor\n'
	@printf '  make bootstrap DRY_RUN=1\n'
	@printf '  make recommend TASK=agents CONTEXT=32768 EXPLAIN=1\n'
	@printf '  make recommend ENGINE=vllm HARDWARE=examples/hardware/4x-a100-80gb.json\n'
	@printf '  make model-pull MODEL=qwen3-8b:q4_k_m\n'
	@printf '  make serve MODEL=qwen3-8b:q4_k_m CONTEXT=16384 && make serve-test\n'
	@printf '  make serve MODEL=qwen3-32b:awq ENGINE=vllm TP=2 PORT=8001\n'
	@printf "  make serve MODEL=qwen3-32b:awq ENGINE=vllm ENGINE_ARGS='--enable-prefix-caching'\\n"
	@printf '  make harness-install NAME=qwen-code\n'
	@printf '  make setup PROFILE=coding-agent DRY_RUN=1\n\n'

version: $(VENV_STAMP) ## Показать версию setup-work
	@$(CLI) version

doctor: $(VENV_STAMP) ## Диагностика системы, инструментов, железа и движков (только чтение)
	@$(CLI) doctor

hardware: $(VENV_STAMP) ## Показать обнаруженное железо
	@$(CLI) hardware

hardware-json: $(VENV_STAMP) ## Отчёт о железе в JSON (для автоматизации)
	@$(CLI) hardware --json

config-show: $(VENV_STAMP) ## Показать итоговую конфигурацию
	@$(CLI) config show

config-validate: $(VENV_STAMP) ## Проверить всю YAML-конфигурацию
	@$(CLI) config validate

##@ Bootstrap (в user-space, идемпотентно; DRY_RUN=1 — предпросмотр, ALLOW_SUDO=1 — системные пакеты)
bootstrap: ## Установить uv + окружение проекта, затем все недостающие инструменты
	@sh scripts/bootstrap.sh
	@$(CLI) bootstrap all

bootstrap-python: ## Установить uv и Python под управлением uv
	@sh scripts/bootstrap.sh
	@$(CLI) bootstrap python

bootstrap-node: $(VENV_STAMP) ## Установить зафиксированные версии Node.js + npm в ~/.local
	@$(CLI) bootstrap node

bootstrap-tools: $(VENV_STAMP) ## Установить git, git-lfs, curl, wget и инструменты сборки
	@$(CLI) bootstrap tools

engine-install: $(VENV_STAMP) ## Установить движок из корпоративного артефакта: ENGINE=llama.cpp|vllm
	$(call require,ENGINE,ENGINE,llama.cpp)
	@$(CLI) engine install

engine-status: $(VENV_STAMP) ## Показать установленные/поддерживаемые движки
	@$(CLI) engine status

##@ Корпоративное (конфиг команды: SETUP_WORK_TEAM_CONFIG_DIR, см. examples/corp/)
corp-setup: $(VENV_STAMP) ## Запустить корпоративные скрипты настройки (pip.conf, .npmrc, ...) с подтверждением
	@$(CLI) corp setup

corp-registries: ## Записать корпоративные PyPI/npm в pip.conf и .npmrc (SETUP_WORK_PYPI_*/NPM_* из bootstrap.env)
	@sh scripts/bootstrap.sh --registries-only

zip-join: ## Склеить части архива (.zip, .z01, .z02, ...) после подтверждения: ZIP=префикс|файл.zip|папка [OUT= ZIP_LAST=1]
	@if [ -z "$(ZIP)" ]; then \
		echo "error: ZIP is required, e.g. make $@ ZIP=/data/Qwen3.8-Flash" >&2; exit 2; fi
	@python3 scripts/join_zip.py "$(ZIP)" $(if $(OUT),-o "$(OUT)") $(if $(filter 1,$(ZIP_LAST)),--zip-last)

corp-status: $(VENV_STAMP) ## HDFS-клиент, Kerberos-тикет, зеркала, настроенные артефакты
	@$(CLI) corp status

corp-env: $(VENV_STAMP) ## Вывести строки export для корпоративного окружения
	@$(CLI) corp env

##@ Модели (ничего не загружается, кроме model-pull)
models: $(VENV_STAMP) ## Каталог моделей [TASK=]
	@$(CLI) models

recommend: $(VENV_STAMP) ## Подобрать модели [TASK= CONTEXT= ENGINE= CONCURRENCY= TP= EXPLAIN=1]
	@$(CLI) recommend

model-pull: $(VENV_STAMP) ## Получить веса из корпоративного артефакта: MODEL=id:variant
	$(call require,MODEL,MODEL,qwen3-8b:q4_k_m)
	@$(CLI) model pull

model-list: $(VENV_STAMP) ## Список распакованных моделей
	@$(CLI) model list

model-info: $(VENV_STAMP) ## Метаданные модели: MODEL=id[:variant]
	$(call require,MODEL,MODEL,qwen3-8b)
	@$(CLI) model info

model-remove: $(VENV_STAMP) ## Удалить распакованную модель: MODEL=id:variant
	$(call require,MODEL,MODEL,qwen3-8b:q4_k_m)
	@$(CLI) model remove

##@ Харнессы
harness-list: $(VENV_STAMP) ## Рецепты харнессов и статус установки
	@$(CLI) harness list

harness-install: $(VENV_STAMP) ## Установить харнесс: NAME= [VERSION= REF=]
	$(call require,NAME,NAME,qwen-code)
	@$(CLI) harness install

harness-update: $(VENV_STAMP) ## Обновить харнесс: NAME= [VERSION= REF=]
	$(call require,NAME,NAME,qwen-code)
	@$(CLI) harness update

harness-remove: $(VENV_STAMP) ## Удалить харнесс: NAME=
	$(call require,NAME,NAME,qwen-code)
	@$(CLI) harness remove

harness-doctor: $(VENV_STAMP) ## Проверить харнесс (все установленные, если NAME пуст)
	@$(CLI) harness doctor

##@ Инференс (по умолчанию слушает 127.0.0.1)
serve: $(VENV_STAMP) ## Запустить сервер: MODEL= [ENGINE= CONTEXT= PORT= TP= ENGINE_ARGS= FOREGROUND=1 PLAN=1]
	$(call require,MODEL,MODEL,qwen3-8b:q4_k_m)
	@$(CLI) serve

serve-auto: $(VENV_STAMP) ## Запустить лучшую скачанную модель для TASK= [ENGINE= CONTEXT= ENGINE_ARGS=]
	@$(CLI) server auto

serve-stop: $(VENV_STAMP) ## Остановить сервер [INSTANCE=default]
	@$(CLI) server stop

serve-status: $(VENV_STAMP) ## Статус сервера
	@$(CLI) server status

serve-logs: $(VENV_STAMP) ## Логи сервера [LINES=100 FOLLOW=1]
	@$(CLI) server logs

serve-test: $(VENV_STAMP) ## Отправить минимальный OpenAI-совместимый запрос [URL=]
	@$(CLI) server test

##@ Профили
setup: $(VENV_STAMP) ## Применить профиль: PROFILE=minimal|developer|coding-agent|local-llm|gpu-inference|full
	@$(CLI) setup

profiles: $(VENV_STAMP) ## Список профилей
	@$(CLI) profile list

profile-show: $(VENV_STAMP) ## Показать итоговый профиль: PROFILE=
	$(call require,PROFILE,PROFILE,developer)
	@$(CLI) profile show

##@ Разработка
sync: $(VENV_STAMP) ## Создать/обновить virtualenv проекта из uv.lock

$(VENV_STAMP): uv.lock pyproject.toml
	@if ! command -v "$(UV)" >/dev/null 2>&1 && [ ! -x "$(UV)" ]; then \
		echo "error: uv not found. Run: make bootstrap   (or sh scripts/bootstrap.sh)" >&2; exit 3; fi
	@$(UV) sync $(if $(UV_DEFAULT_INDEX)$(UV_INDEX_URL),,--frozen) --quiet
	@touch $@

test: $(VENV_STAMP) ## Запустить тесты
	@$(UV) run --frozen pytest

lint: $(VENV_STAMP) ## Линтер и проверка форматирования
	@$(UV) run --frozen ruff check .
	@$(UV) run --frozen ruff format --check .

format: $(VENV_STAMP) ## Отформатировать код и применить безопасные исправления
	@$(UV) run --frozen ruff format .
	@$(UV) run --frozen ruff check --fix .

check: lint test config-validate ## Линтер + тесты + проверка конфигурации

clean: ## Удалить артефакты сборки и кэши установщика setup-work
	@rm -rf .pytest_cache .ruff_cache build dist
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@if [ -f "$(VENV_STAMP)" ]; then $(CLI) clean; fi

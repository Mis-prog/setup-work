# setup-work

Последовательная подготовка рабочего места в корпоративной сети:
**инфраструктура → агенты и GigaChat → дополнительная локальная модель**.

## Начало работы

Нужны Git, Make, Bash и установленный Python 3.11+ с модулями `venv` и `pip`.
Для HDFS нужны клиент Hadoop и, если требуется кластером, билет Kerberos.
Драйверы GPU устанавливаются отдельно. Репозиторий, харнессы и их настройки храните на постоянном томе, например NFS.

```bash
export SETUP_WORK_HOME="$HOME/nfs/setup-work"
export SETUP_WORK_USER_CONFIG_DIR="$PWD/config"
make init
. "$SETUP_WORK_HOME/env.sh"
```

Заполните адреса реестров, пути к файлам токенов и внутренние репозитории в `config/`.
`CHANGE-ME` — заготовки, которые нужно заменить своими значениями.

```bash
make setup
# Если системный python3 старый:
make setup PYTHON=python3.12
```

## Что делает make setup

1. Настраивает корпоративные pip/npm и создаёт `.venv` проекта. Устанавливает
   закреплённые Node.js/npm. По желанию устанавливает uv. Загружает настроенные JAR.
2. Спрашивает, нужны ли агенты. Для выбранных агентов по очереди забирает внутренний
   Git-репозиторий и запускает его `setup.sh` в корне checkout через Bash.
   Скрипт репозитория отвечает за установку агента, настройку и запуск GigaChat API.
3. Проверяет GPU. Без доступных GPU локальная установка пропускается. При наличии
   GPU предлагает дополнительно установить движок, загрузить подходящую модель
   и запустить её API. GigaChat остаётся основным режимом агентов.

При ошибке обязательного шага выполнение останавливается. Следующий запуск
переиспользует харнессы и успешно завершённые установки в текущем контейнере.
Веса моделей и движки по умолчанию находятся во временном каталоге контейнера;
после его пересоздания устанавливаются заново. Внутри
`setup.sh` возобновление и повторный запуск должны поддерживаться самим репозиторием.
Промежуточный результат хранится в `$SETUP_WORK_HOME/state/setup-state.json`.

Выбор можно задать заранее:

```bash
make setup UV=1 AGENTS=1 HARNESSES=qwen-code,hermes LOCAL=0
make setup AGENTS=0 LOCAL=1
make setup AGENTS=1 UPDATES=0 LOCAL=0
make setup DRY_RUN=1 AGENTS=1 HARNESSES=qwen-code LOCAL=0
```

`AGENTS` и `LOCAL`: `1` — включить, `0` — пропустить; без значения — спросить.
`HARNESSES` — имена через запятую в нужном порядке. Без списка берутся Git-рецепты
с `install.method: setup` в порядке YAML. `UPDATES=0` отключает проверку обновлений.
`YES=1` явно подтверждает все вопросы, включая обновления; для автоматизации лучше
также указать `AGENTS` и `LOCAL`. `DRY_RUN=1` показывает действия без установки,
загрузок и изменения состояния; незаданные ответы показываются как будущие вопросы.

## Метрики использования агентов и HDFS

Сбор выключен по умолчанию. В `config/corporate.yaml` задайте папку отдела:

```yaml
metrics:
  enabled: true
  hdfs_uri: hdfs:///ml/metrics/agent-usage
  department: data-engineering
  # user_id: employee-123  # иначе логин ОС; задавайте отдельно на каждом рабочем месте
  upload_interval_s: 300
```

Сначала проверьте доступ пользователя к этой папке и билет Kerberos (`kinit`).
Выгрузка использует существующие `hdfs.cli`, `hdfs.extra_args`, `hdfs.kerberos`.
Токены берутся из `usage` ответов API: это учёт запросов, включая шаги агента,
инструментальные циклы и реальные повторные обращения к модели.

### Автоматический сбор через локальный API-прокси

Для агента с HTTP API Chat Completions, Responses или Messages запустите:

```bash
make metrics-proxy METRICS_AGENT=qwen-code METRICS_UPSTREAM=http://127.0.0.1:8000/v1
```

`METRICS_UPSTREAM` — текущий адрес модели или **уже настроенного GigaChat-адаптера**
из репозитория агента. Прокси пересылает заголовки авторизации и ответы, включая SSE.
Прокси не получает и не обновляет OAuth-токены самостоятельно. Для корпоративного
HTTPS CA можно задать `SSL_CERT_FILE`.

В настройках **самого агента** замените адрес API на `http://127.0.0.1:8766/v1`.
Если агент читает стандартную переменную, пример запуска в другом терминале:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8766/v1
# Запустите qwen-code его обычной командой.
```

Без изменения адреса API запросы проходят мимо сборщика и не учитываются.
Настройки внутренних форков находятся в их `setup.sh`; setup-work не может
автоматически определить их адреса, формат конфигурации и способ авторизации.
Для каждого агента нужен отдельный экземпляр прокси с его именем и портом
(`METRICS_PORT=8767` и т. д.). Так запросы правильно относятся к агентам.
Слушается только `127.0.0.1`. Прокси работает в текущем терминале до Ctrl+C;
для постоянной работы включите команду в принятый в отделе менеджер процессов.
Сбор начинается с момента подключения; предыдущая история сессий не импортируется.

Прокси сохраняет одну запись на POST к `/chat/completions`, `/completions`,
`/responses` или `/messages`. GET `/models` и проверки доступности не считаются.
В streaming Chat Completions добавляется `stream_options.include_usage=true`.
Если адаптер не принимает этот параметр, запускайте CLI с `--no-include-usage`;
для полного учёта адаптер всё равно должен возвращать финальный `usage`.
Прокси не повторяет запросы к модели при ошибках.

Пока прокси запущен, очередь автоматически выгружается каждые `upload_interval_s`.
Вручную проверить или выгрузить её можно так:

```bash
make metrics-status
make metrics-status JSON=1
make metrics-upload
make metrics-upload DRY_RUN=1
```

При недоступном HDFS, истёкшем Kerberos-билете или завершении процесса записи
сохраняются в `<state_dir>/metrics/usage.sqlite3`. После восстановления доступа
выгрузка повторяется. Храните `SETUP_WORK_HOME` на постоянном томе. SQLite требует
работающих файловых блокировок; для NFS это необходимо проверить при развёртывании.
Перед завершением работы можно выполнить `make metrics-upload`.

### Интеграция через hook API-адаптера

Если агент нельзя направить через прокси, его API-адаптер может записывать метрики
после каждого запроса. Пример из Python в окружении, где установлен setup-work:

```python
from setup_work.config import Config
from setup_work.metrics import record_usage

record_usage(
    Config(),
    agent="hermes",
    model=response["model"],
    request_id=local_request_id,  # свой уникальный ID реального вызова API
    usage=response.get("usage"),
    session_id=session_id,  # необязательно
)
```

Либо вызов CLI без зависимости от Python-окружения агента:

```bash
.venv/bin/python -m setup_work metrics record \
  --agent hermes --model GigaChat --request-id request-uuid \
  --input-tokens 1500 --output-tokens 300 --cached-input-tokens 500

# JSON ответа API можно передать через stdin: --response-file -
.venv/bin/python -m setup_work metrics record \
  --agent hermes --model GigaChat --request-id request-uuid \
  --response-file /path/to/response.json
```

Выбирайте один способ сбора для одного запроса, чтобы hook и прокси не учитывали
его дважды. При повторе **записи метрик** сохраняйте `request_id`: повтор отбрасывается.
Новый реальный вызов API, даже с тем же текстом, должен получать новый ID.
Передавайте финальный расход отдельного запроса; накопительный расход всей сессии
и промежуточные SSE-счётчики передавать нельзя. При записи исторических событий
есть `--timestamp` в ISO 8601 с часовым поясом. Без `usage` счётчики будут `null`.

Внутренним `setup.sh` и `install.start` передаются `SETUP_WORK_METRICS_ENABLED`,
`SETUP_WORK_METRICS_AGENT`, `SETUP_WORK_METRICS_PYTHON` (Python setup-work) и
`SETUP_WORK_METRICS_STATE_DIR`. Скрипт форка должен сохранить нужные параметры
в своей конфигурации/команде запуска. Для hook вызов выглядит как
`"$SETUP_WORK_METRICS_PYTHON" -m setup_work metrics record ...`.
При использовании только hook настройте периодический `metrics upload` через
планировщик отдела; автоматический таймер работает внутри `metrics proxy`.

### Формат для агрегации

```text
hdfs:///ml/metrics/agent-usage/
  dt=2026-10-20/device_id=<device-id>/<batch-id>.jsonl
```

Одна строка JSON — один запрос API. Поля версии 1: `schema_version`, `event_id`,
`timestamp` (UTC), `device_id`, `user_id`, `department`, `agent`, `model`, `request_id`,
`session_id`, `http_status`, `outcome`, `duration_ms`, `usage_status`, `input_tokens`,
`output_tokens`, `total_tokens`, `cached_input_tokens`, `cache_creation_input_tokens`,
`reasoning_tokens`. `outcome`: `ok`, `error`, `interrupted`; `usage_status`: `reported`,
`missing`. Нет текстов запросов/ответов, файлов проекта, ключей и заголовков авторизации.
`user_id` по умолчанию — логин ОС; `device_id` автоматически выводится из ID установки
и hostname. Для нескольких пользователей не используйте общую учётную запись ОС;
при необходимости задавайте `user_id` и `device_id` отдельно на каждом компьютере.

`input_tokens` включает cache read/write; `output_tokens` включает reasoning;
`total_tokens = input_tokens + output_tokens`. Cache и reasoning — вложенные
счётчики, **не прибавляйте их к total повторно**. Нормализуются поля OpenAI-совместимых
API, GigaChat (`precached_prompt_tokens`) и Anthropic. Это число токенов, а не
стоимость: тарифы и скидки за cache вычисляются отдельно по модели/провайдеру.
Если API не вернул usage, поток оборвался или usage оказался некорректным,
запрос может иметь `usage_status=missing` и `null` вместо счётчиков. Если usage уже
получен до обрыва, он сохраняется вместе с `outcome=interrupted`. Суммы отражают
известный расход; рядом обязательно считайте число `missing` и долю покрытия.

Пакеты ограничены `batch_size` (по умолчанию 1000), разделены по дате UTC и устройству.
Сначала записывается скрытый `.pending`, затем выполняется rename в `.jsonl`.
При повторе содержимое и имя пакета сохраняются; уже опубликованный пакет
проверяется и подтверждается без создания копии. При смене `hdfs_uri` незавершённые
пакеты дописываются в прежний адрес, новые используют новый. Выгруженные записи
локально сохраняются для просмотра и дедупликации; автоматической очистки пока нет.

Пример агрегации в Spark, после накопления нескольких пакетов:

```python
from pyspark.sql import functions as F

raw = spark.read.option("basePath", "hdfs:///ml/metrics/agent-usage").json(
    "hdfs:///ml/metrics/agent-usage/dt=*/device_id=*/*.jsonl"
)
events = raw.filter(F.col("schema_version") == 1).dropDuplicates(["event_id"])
metrics = events.groupBy("dt", "department", "user_id", "agent", "model").agg(
    F.count("event_id").alias("requests"),
    F.sum("total_tokens").alias("known_total_tokens"),
    F.sum("input_tokens").alias("known_input_tokens"),
    F.sum("output_tokens").alias("known_output_tokens"),
    F.sum(F.when(F.col("usage_status") == "missing", 1).otherwise(0)).alias("missing_usage"),
)
```

## Конфигурация

| Файл | Назначение |
|---|---|
| `config/bootstrap.env` | Адреса PyPI/npm, файлы токенов, CA и прокси |
| `config/defaults.yaml` | Версии инструментов и параметры сценария `setup` |
| `config/harnesses.yaml` | Внутренние Git-репозитории агентов и их `setup.sh` |
| `config/corporate.yaml` | Источники JAR, моделей, движков и метрики HDFS |
| `config/models.yaml` | Каталог моделей и параметры для оценки памяти |

`SETUP_WORK_USER_CONFIG_DIR` выбирает каталог настроек. Без него используются
`$SETUP_WORK_HOME/config`, затем каталог XDG `~/.config/setup-work`.
Встроенные YAML из `config/` служат базовым слоем; пользовательские значения имеют приоритет.

### Обновление через rsync с сохранением внутренних настроек

Если на целевой машине заполнены `config/` и уже есть `.git`, обновляйте только
код, исключив эти каталоги. Сначала посмотрите список изменений:

```bash
rsync -av --dry-run --itemize-changes \
  --exclude='/.git/' \
  --exclude='/config/' \
  --exclude='/examples/corp/' \
  --exclude='/.venv/' \
  --exclude='/.env' --exclude='/.env.*' \
  --exclude='__pycache__/' \
  --exclude='/.pytest_cache/' --exclude='/.ruff_cache/' \
  --exclude='/build/' --exclude='/dist/' \
  /path/to/new/setup-work/ user@host:/path/to/existing/setup-work/
```

Замените пути и адрес машины своими, затем повторите команду без `--dry-run`.
Для копирования на той же машине последний аргумент может быть обычным путём.
Завершающий `/` у источника означает копирование содержимого каталога.
Не добавляйте `--delete`: этот пример сохраняет существующие файлы, включая
удалённые из новой версии. Старые файлы, если они мешают обновлению, нужно
разбирать отдельно. Если вы изменяли другие файлы кода, rsync их перезапишет;
сохраните такие изменения перед обновлением.

На целевой машине `.git` и его история останутся прежними:

```bash
git status --short
git diff -- config/
git diff -- src/ Makefile pyproject.toml
make sync
make config-validate
```

`git diff` показывает изменения относительно прежнего коммита целевой копии;
новые файлы видны в `git status` как untracked. Тег из исходной копии через такой
rsync не переносится: Git-метаданные исключены. Новые параметры конфигурации,
например секцию `metrics` версии 4.5.0, добавляйте в сохранённые YAML вручную,
сверяясь с README или шаблонами исходной копии.

Для дальнейших обновлений удобнее хранить заполненные конфиги вне репозитория,
например в `$SETUP_WORK_HOME/config`, и указывать этот путь через
`SETUP_WORK_USER_CONFIG_DIR`. Тогда внутренние адреса, реестры и параметры HDFS
сохраняются отдельно от обновляемого кода и шаблонов `config/`.

### Реестры, токены и uv

В `bootstrap.env` используются строки `KEY=VALUE`, без исполнения shell-команд.
Уже заданные переменные окружения имеют приоритет. Секреты храните в отдельных
файлах, указанных в `SETUP_WORK_PYPI_AUTH_FILE` и `SETUP_WORK_NPM_AUTH_FILE`.
Пути поддерживают `~/` и `$HOME/`. Для реестра без авторизации оставьте `AUTH_FILE` пустым.

```bash
make registries DRY_RUN=1
make registries
```

Создаются `pip.conf`, `uv.toml` и `npmrc` с правами `600` в
`$SETUP_WORK_HOME/registries/`. Они содержат учётные данные. При смене токена повторите
`make registries`. Для своего CA задайте `PIP_CERT`, `SSL_CERT_FILE` и
`SETUP_WORK_NPM_CAFILE` в `bootstrap.env`.

uv необязателен: `make setup UV=1` или `setup.uv: true` в `defaults.yaml`.
Он устанавливается через pip из корпоративного PyPI в отдельное окружение, а
команды `uv` и `uvx` становятся доступны в `bin/`. Для pip и uv задаётся один PyPI,
автоматическая загрузка Python через uv отключена. Сам проект использует
`python -m venv`, pip и `.venv/bin/python`, без обязательной зависимости от uv.
Версия uv закреплена в `tools.uv.version`.

### Агенты и обновления

```yaml
harnesses:
  qwen-code:
    source: git
    repository: https://bitbucket.example/scm/team/qwen-code.git
    ref: main
    install:
      method: setup
      script: setup.sh
      args: []
    # start: [bash, scripts/start-api.sh]  # только если такая команда есть в репозитории
```

Git использует существующую аутентификацию. Во время загрузки интерактивный запрос
пароля Git отключён — заранее проверьте credential helper и доступ к репозиторию.
Сам `setup.sh` запускается с доступом к терминалу, чтобы пройти его настройку.

Скрипт получает `SETUP_WORK_BACKEND=gigachat` и наследует окружение пользователя,
настройки реестров и `PATH`. Реальные параметры GigaChat, авторизация и запуск
прокси остаются в скрипте каждого репозитория; единого формата для них проект
не навязывает. `SETUP_WORK_BACKEND` — подсказка скрипту, её поддержку нужно учесть
внутри репозитория. Скрипт также получает `SETUP_WORK_HARNESS_HOME`,
`SETUP_WORK_HARNESS_CONFIG` и `SETUP_WORK_HARNESS_BIN` — постоянные каталоги харнесса.
Он должен сохранять окружение и настройки в checkout или этих каталогах;
файлы, записанные только в файловую систему контейнера, проект автоматически не переносит.

Установленный коммит и рецепт сохраняются. При повторном запуске проверяется
новая ревизия указанной ветки или тега; для фиксированного коммита используется он.
Перед обновлением показываются текущая и доступная ревизии и задаётся вопрос.
Отказ оставляет установленную версию. `setup.sh` актуального харнесса повторно
не запускается. Если после пересоздания контейнера нужно поднять GigaChat-прокси,
задайте `install.start` — реальную команду запуска уже установленного API.
Она выполняется при повторном использовании харнесса, без переустановки.
Без этой настройки автоматический повторный запуск API не обещается. Локальные изменения в отслеживаемых файлах
не перезаписываются. Ошибка `setup.sh` не помечает новую установку успешной.

### JAR для Trino и Iceberg

Источники пока не выбраны, поэтому по умолчанию:

```yaml
jars:
  trino: null
  iceberg: null
```

Эти шаги пропускаются. Когда источник определится, укажите прямой файл:

```yaml
jars:
  trino:
    uri: hdfs:///ml/jars/trino-jdbc-VERSION.jar
    format: file
    version: "VERSION"
```

Поддерживаются HDFS, прямой HTTPS URL (в том числе файл из Bitbucket) и локальный
абсолютный путь. Авторизация HTTPS использует настройки среды/URL-клиента;
специальный способ входа в Bitbucket для JAR пока не задан. Файлы сохраняются в
`$SETUP_WORK_HOME/data/tools/jars/<имя>/`; подключение драйвера к конкретному
приложению выполняется в его настройках. Java устанавливается отдельно.
Для новой сборки измените URI или `version`.

### Дополнительная локальная модель

В `corporate.yaml` укажите источники готовых движков и моделей. Модели vLLM
используют полный каталог весов/токенизатора или `zip-parts` с комплектом
`<имя>-1-of-N.zip`; GGUF — прямой файл. Движки — готовые корпоративные архивы:
llama.cpp с `llama-server` и библиотеками, vLLM с переносимым venv.
ОС, CUDA и версия Python должны подходить установленному движку.

Выбор учитывает совместимость, доступную память GPU, контекст и число устройств.
Порядок ранжирования стабилен для одинаковой конфигурации и профиля оборудования.
Подбираются только варианты с заданными источниками модели и движка.
Оценка памяти приблизительная; особенности гибридного внимания Qwen описаны
в [описании моделей](examples/corp/qwen-27b/README.md).

```bash
make recommend TASK=coding EXPLAIN=1
make setup AGENTS=0 LOCAL=1 MODEL=qwen3.8-27b:awq ENGINE=vllm CONTEXT=8192
make serve-status INSTANCE=local
make serve-logs INSTANCE=local
make serve-stop INSTANCE=local
```

Веса, движки, временные загрузки и состояние локального сервера находятся в
`$SETUP_WORK_RUNTIME_DIR`; по умолчанию это временный каталог `setup-work-<uid>`
внутри контейнера. Его путь виден в `make config-show`. При необходимости задайте
`SETUP_WORK_RUNTIME_DIR` через `export` в shell перед всеми командами проекта.

Сервер запускается под именем `local`. Он не изменяет настройки GigaChat агентов.
Порт можно настроить в `inference.port`; по умолчанию `127.0.0.1:8000`.
Настройкой `setup.start_local: false` можно отключить запуск, оставив установку.

## Новый терминал и bashrc

```bash
. "$HOME/nfs/setup-work/env.sh"
export SETUP_WORK_USER_CONFIG_DIR="$PWD/config"
```

`env.sh` подключает `PATH`, пути к настройкам/кэшам и сохранённый каталог конфигурации.
Харнессы и настройки сохраняются в `$SETUP_WORK_HOME`; монтируйте его по одному
и тому же пути. В контейнере нужны совместимые Python и системные библиотеки.
Команды из `data/harnesses/*/bin` также добавляются в `PATH`. Его можно подключать из
`~/.bashrc` для Bash или `~/.zshrc` для Zsh. Эти файлы содержат команды, выполняемые
при запуске интерактивной оболочки. Проект не редактирует их автоматически.

## Проверки и отдельные команды

```bash
make help
make doctor
make config-validate
make sync                 # зависимости проекта через pip
make check                # ruff, pytest, конфигурация
```

Отдельные команды для моделей, движков, харнессов и сервера доступны через `make help`.
Серверные тесты требуют возможности открыть локальный порт. Тестам не нужны GPU,
HDFS или доступ к внутренним репозиториям.

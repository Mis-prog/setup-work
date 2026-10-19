# setup-work

Подготовка рабочего места для AI/ML в корпоративной сети: настройка PyPI/npm,
установка инструментов и агентов, загрузка моделей из HDFS и запуск локального
OpenAI-совместимого API через llama.cpp или vLLM.

В этом репозитории настройки редактируются **прямо в `config/`**. Адреса с
`CHANGE-ME` и пути HDFS — заготовки: замените их своими. Параметры трёх
моделей Qwen уже заполнены по указанным источникам Hugging Face.

## 1. Что подготовить

На целевой машине должны быть:

- `git`, `make`, `python3` версии 3.11 или новее;
- модуль `pip`, если uv будет устанавливаться из корпоративного PyPI;
- доступ к корпоративным PyPI, npm и Bitbucket;
- для HDFS — настроенный клиент Hadoop, конфигурация вашего кластера и доступ к нему;
- для Kerberos — `kinit` и `klist`;
- для vLLM — подходящая Linux-машина с NVIDIA GPU, драйвером и установленным Python
  той же основной и дополнительной версии, что использовалась при сборке venv;
- постоянный том, например NFS, с местом под архивы, распакованные модели и движки.

Проверьте инструменты:

```bash
python3 --version
python3 -m pip --version
git --version
make --version
hdfs version
```

Python проект автоматически не скачивает. Клиенты HDFS, Kerberos, драйвер NVIDIA
и Bun для рецепта opencode тоже устанавливаются отдельно. Архивы движков нужно
заранее подготовить под ОС, архитектуру и окружение целевой машины.

## 2. Выбрать каталог данных и подключить настройки

Откройте корень этого репозитория, где находятся `Makefile` и каталог `config/`.
Сам репозиторий тоже разместите на постоянном томе: его `.venv` хранится рядом
с `pyproject.toml`.

```bash
# Постоянный каталог инструментов, моделей, кэшей и состояния.
export SETUP_WORK_HOME="$HOME/nfs/setup-work"

# Читать настройки прямо из config/ текущего репозитория.
export SETUP_WORK_USER_CONFIG_DIR="$PWD/config"

# Создать env.sh; существующие файлы настроек сохраняются.
make init
. "$SETUP_WORK_HOME/env.sh"
```

`SETUP_WORK_HOME` и каталог репозитория могут находиться в разных местах.
Имена файлов и их назначение:

| Файл в репозитории | Что нужно заполнить |
|---|---|
| `config/bootstrap.env` | URL PyPI/npm, логин PyPI, пути к токенам, источник uv, при необходимости CA и прокси |
| `config/corporate.yaml` | HDFS-пути к моделям, движкам и инструментам; пути внутри ZIP; хост Bitbucket |
| `config/models.yaml` | Параметры архитектуры, типы и размеры весов ваших моделей |
| `config/harnesses.yaml` | URL внутренних репозиториев агентов, тег/коммит, команды сборки и пути к исполняемым файлам |
| `config/defaults.yaml` | Версии инструментов, контрольные суммы, порт, лимиты памяти и другие общие настройки |

Без `SETUP_WORK_USER_CONFIG_DIR` пользовательские настройки берутся из
`$SETUP_WORK_HOME/config`, а без `SETUP_WORK_HOME` — из `~/.config/setup-work/`
с учётом переменных XDG. Пользовательские YAML-настройки имеют приоритет над
`config/` репозитория. Явный `export` выше выбирает нужный каталог и для bootstrap.

## 3. Заполнить PyPI, npm и файлы с токенами

Отредактируйте **`config/bootstrap.env`**:

```sh
SETUP_WORK_PYPI_INDEX=https://CHANGE-ME-repo-host/repository/pypi-group/simple
SETUP_WORK_PYPI_USER=token
SETUP_WORK_PYPI_AUTH_FILE=~/nfs/tokens/pypi-token.txt

SETUP_WORK_NPM_REGISTRY=https://CHANGE-ME-repo-host/repository/npm-group/
SETUP_WORK_NPM_AUTH_FILE=~/nfs/tokens/npm-token.txt
SETUP_WORK_NPM_STRICT_SSL=true
SETUP_WORK_NPM_OPTIONS="audit=false fetch-retries=5 save-exact=true"

SETUP_WORK_UV_FROM_PIP=1
```

У администратора реестра уточните реальные URL и логин PyPI. Если реестр не требует
аутентификации, оставьте соответствующий `AUTH_FILE` пустым, например:

```sh
SETUP_WORK_PYPI_AUTH_FILE=
```

Создайте отдельные файлы с выданными токенами по указанным путям:

```bash
mkdir -p "$HOME/nfs/tokens"
(umask 077; touch "$HOME/nfs/tokens/pypi-token.txt" "$HOME/nfs/tokens/npm-token.txt")
chmod 600 "$HOME/nfs/tokens/pypi-token.txt" "$HOME/nfs/tokens/npm-token.txt"
```

Откройте каждый файл в редакторе и вставьте только его токен. Для npm требуется
готовое значение `_authToken`; скрипт читает его как есть, без преобразования
в Base64. Токены не вставляются в YAML, `bootstrap.env`, URL или команды shell.
Сгенерированные `pip.conf`, `uv.toml` и `npmrc` содержат учётные данные и создаются
с правами `600`; не публикуйте их содержимое.

`bootstrap.env` содержит строки `KEY=VALUE`. Этот файл не исполняется как shell:
`$(команда)` и подстановка произвольных переменных не работают. Пути к токенам
поддерживают `~/` и префикс `$HOME/`. Уже заданные переменные окружения имеют
приоритет над значениями файла.

Если uv ещё не установлен, `SETUP_WORK_UV_FROM_PIP=1` устанавливает закреплённую
проектом версию из вашего PyPI. Альтернатива — заполнить
`SETUP_WORK_UV_ARCHIVE` путём к архиву на HDFS, NFS или внутреннем HTTPS.

### Корпоративные сертификаты и прокси

Если нужен собственный CA, раскомментируйте и поправьте в `bootstrap.env`:

```sh
PIP_CERT=/etc/pki/tls/certs/corp-ca-bundle.crt
SSL_CERT_FILE=/etc/pki/tls/certs/corp-ca-bundle.crt
SETUP_WORK_NPM_CAFILE=/etc/pki/tls/certs/corp-ca-bundle.crt
```

Для последующих Python-команд аналогичные несекретные переменные можно добавить
в `env` файла `corporate.yaml`. Для Git CA настраивается отдельно после подключения
`env.sh`:

```bash
git config --global http.sslCAInfo /путь/к/corp-ca.crt
```

Если нужен прокси, заполните `HTTP_PROXY`, `HTTPS_PROXY` и `NO_PROXY` в
`bootstrap.env`. Переменные для текущего shell и команд, запущенных вне bootstrap,
задавайте также через `export` либо через подходящий раздел `corporate.yaml`.
`SETUP_WORK_PYPI_TRUSTED_HOST` отключает проверку TLS для указанного хоста,
а `SETUP_WORK_NPM_STRICT_SSL=false` — для npm; для доверенного CA используйте
пути к сертификатам.

## 4. Прописать модели и движки на HDFS

Отредактируйте **`config/corporate.yaml`**. В нём уже есть заготовки:

```yaml
models:
  "qwen3.8-27b:full":
    uri: hdfs:///ml/models/hf/qwen3.8-27b/
    format: zip-parts
  "qwen3.8-27b:awq":
    uri: hdfs:///ml/models/hf/qwen3.8-27b-awq/
    format: zip-parts
  "qwen3.6-27b:q4_k_m":
    uri: hdfs:///ml/models/gguf/Qwen3.6-27B-Q4_K_M.gguf

engines:
  llama.cpp:
    uri: hdfs:///ml/engines/llama.cpp.zip
    path: llama.cpp/build/bin/llama-server
  vllm:
    uri: hdfs:///ml/engines/vllm.zip
    path: vllm-venv
```

`uri` — папка с частями весов vLLM, отдельный GGUF-файл или архив движка. `path` — путь **внутри ZIP**, относительно его корня:
для llama.cpp это файл `llama-server`, для vLLM — каталог с `pyvenv.cfg`.
Если в корне ZIP сразу лежат `src/` и `build/`, укажите
`path: build/bin/llama-server`.

| Артефакт | Что должно быть в архиве |
|---|---|
| GGUF | Обычный файл `.gguf` по прямому HDFS URI; архивировать его не нужно |
| Модель vLLM | Полный каталог с `config.json`, весами `*.safetensors`, индексом шардов и всеми файлами токенизатора |
| llama.cpp | Готовый `llama-server` и необходимые библиотеки; исходники и `build/` могут оставаться в архиве |
| vLLM | Установленное venv со всеми зависимостями, каталогами `bin/`, `lib/` и файлом `pyvenv.cfg` |
| uv | Архив под версию и SHA256 из `config/defaults.yaml` |

Установщик llama.cpp не компилирует исходники. venv vLLM переносится с исправлением
путей к Python и строк запуска скриптов; совместимость ОС, Python, CUDA и библиотек
нужно обеспечить при подготовке архива. Для сохранения ссылок окружения используйте
`zip -r -y`. Модели должны содержать сами веса, а не ссылки на чужой кэш.

Для vLLM укажите **папку HDFS** и `format: zip-parts`. Внутри должен находиться
один комплект файлов, например `Qwen_Qwen3.8-27B-1-of-19.zip`,
`Qwen_Qwen3.8-27B-2-of-19.zip`, …, `Qwen_Qwen3.8-27B-19-of-19.zip`.
Количество берётся из имён, поэтому допустимо и другое число частей. Посторонние
файлы игнорируются; пропуски, повторяющиеся номера и несколько комплектов
вызывают ошибку с подсказкой. Файлы скачиваются в числовом порядке. Если это
самостоятельные ZIP, они распаковываются в общий каталог; если это последовательные
куски одного ZIP, сначала собирается полный архив. После успешной установки
скачанные копии удаляются; `keep_archive: true` оставляет части в кэше.
При ошибке установленная модель сохраняется. Учтите место под части, собранный
ZIP и распакованные веса. Ручной `make zip-join` для этой загрузки не нужен.

В разделе `tools` поправьте путь к архиву uv. Для новых версий обновите версию
и контрольные суммы в `defaults.yaml`; для начальной установки uv соответствующие
параметры находятся и в `scripts/bootstrap.sh` / `bootstrap.env`.

**Node.js и npm устанавливаются из корпоративного npm registry** при `make bootstrap`.
Адрес, файл токена и CA задаются в `config/bootstrap.env`: `SETUP_WORK_NPM_REGISTRY`,
`SETUP_WORK_NPM_AUTH_FILE`, `SETUP_WORK_NPM_CAFILE`. Сначала выполните `make registries`.
Установщик читает созданный npmrc и работает через Python: Node.js/npm заранее не нужны.
В `config/defaults.yaml` закреплены `tools.node.version` и `tools.npm.version`.
Реестр должен содержать `npm` и платформенный пакет Node той же версии:
`node-linux-x64`, `node-linux-arm64`, `node-darwin-x64` либо `node-bin-darwin-arm64`.
Загрузка проверяется по `dist.integrity` (SHA512/SHA256); публичный npm не используется
как запасной источник. В метаданных прокси публичный адрес tarball переписывается
на корпоративный реестр. Установщик создаёт команды `node`, `npm`, `npx` в `bin_dir`.
Поддерживаются Linux с glibc и macOS, x64/arm64. При уже установленном подходящем
Node.js и доступном npm повторная установка пропускается.
Для принудительной установки закреплённых версий: `make bootstrap FORCE=1`.
Если явно задать `tools.node.uri` в `corporate.yaml`, будет использован прежний
режим установки готового архива с проверкой SHA256.

Для отдельного архива или GGUF можно указать `sha256` самого файла.
Для папки `zip-parts` единая SHA256 не задаётся; ZIP проверяет CRC при распаковке:

```bash
sha256sum /путь/к/архиву.zip
```

Для HTTPS контрольная сумма обязательна. Новую сборку публикуйте под новым URI
или меняйте `sha256`: установленный артефакт с прежними URI и SHA256 переиспользуется.

Проверьте доступ к реальным путям:

```bash
kinit
klist -s
hdfs dfs -ls hdfs:///ml/models/
hdfs dfs -ls hdfs:///ml/engines/
```

В `hdfs.cli` можно указать полный путь к клиенту. При обязательном Kerberos
задайте `hdfs.kerberos: required`. Модели и движки скачиваются из HDFS и
распаковываются на диск/NFS; сервер читает распакованные файлы. На время установки
нужно место под архив и распакованное содержимое, при обновлении — также под старую копию.

## 5. Проверить выбранные модели

В **`config/models.yaml`** оставлены только ваши модели:

| Идентификатор для команд | Формат | Движок |
|---|---|---|
| `qwen3.8-27b:full` | BF16 safetensors | vLLM |
| `qwen3.8-27b:awq` | AWQ INT4, активации BF16 | vLLM |
| `qwen3.6-27b:q4_k_m` | GGUF Q4_K_M | llama.cpp |

Параметры уже заполнены по `config.json` и метаданным указанных репозиториев.
Все три источника имеют лицензию Apache-2.0. Размеры файлов весов:

- полная Qwen3.8: **55.563006776 GB**, 18 файлов safetensors;
- AWQ Qwen3.8: **21.018000928 GB**, 5 файлов safetensors;
- выбранный GGUF Qwen3.6: **16.817244384 GB**.

Источники, ревизии и контрольная сумма GGUF записаны в
[описании моделей](examples/corp/qwen-27b/README.md).
Пути к вашим HDFS-архивам и их SHA256 по этим публичным ссылкам определить нельзя:
поправьте их в `corporate.yaml` и резервном `source` варианта модели.

У обеих архитектур 64 слоя: 48 Gated DeltaNet и 16 полного внимания, контекст
262144 токена. Текущая формула памяти setup-work считает все слои обычными KV-слоями,
поэтому оценка для этих моделей приближённая. Фактическую память и поддержку
архитектуры нужно проверить на вашей сборке движка.

AWQ-источник использует `compressed-tensors`. В HDFS-архив включите его собственный
`config.json`, все файлы весов и токенизатора. vLLM определяет метод квантования
из конфигурации модели. Для GGUF укажите прямой путь к `Qwen3.6-27B-Q4_K_M.gguf`.

Для других моделей параметры берутся из `config.json`/`text_config` или метаданных
GGUF: `architecture`, `parameters_b`, `context_length`, `attention`, `dtype`, `size_gb`.
`size_gb` — сумма размеров файлов весов в байтах / 1 000 000 000, без архива.
Для MoE дополнительно нужны `active_parameters_b` и `moe`.

`source` в варианте модели — резервный путь. Если тот же вариант указан
в `corporate.yaml`, используется путь оттуда. Остальные модели удалены из рабочего каталога.

## 6. Заполнить харнессы и доступ к Bitbucket

В **`config/harnesses.yaml`** подготовлены qwen-code, opencode, pi, hermes,
deepseek и skillhub из внутренних Git-репозиториев. Для каждого нужного рецепта:

1. Скопируйте HTTPS-ссылку из кнопки Clone в Bitbucket в `repository`.
2. Укажите ваш тег или коммит в `ref` вместо `CHANGE-ME`.
3. Сверьте `install.build` и пути в `binaries` с вашим форком.
4. Проверьте требования `requires` и `requires_commands`.

```yaml
repository: https://CHANGE-ME-bitbucket-host/scm/CHANGE-ME/qwen-code.git
ref: CHANGE-ME
```

В `corporate.yaml` укажите соответствующий хост без схемы и пути:

```yaml
trusted_git_hosts:
  - CHANGE-ME-bitbucket-host
```

Для этого хоста разрешаются команды сборки из рецепта без `ALLOW_BUILD=1`.
Git использует вашу аутентификацию через credential helper. Настройте её и заранее
проверьте доступ командой `git ls-remote` с реальным URL: при установке интерактивный
запрос Git отключён. После подключения `env.sh` глобальные настройки Git находятся
в `$SETUP_WORK_HOME/gitconfig`.

Для opencode в текущем рецепте отдельно нужен Bun. Python-рецепты ниже используют
Python 3.12. Команды сборки — заготовки, которые нужно сверить с внутренними форками.
`skillhub` только клонируется; раскладка навыков по агентам пока не выполняется.
claude-code, codex и aider сохранены как пакетные рецепты для корпоративных npm/PyPI.

## 7. Применить настройки и проверить рабочее место

После заполнения файлов, из корня репозитория:

```bash
export SETUP_WORK_USER_CONFIG_DIR="$PWD/config"
. "$SETUP_WORK_HOME/env.sh"

# Показать, куда будут записаны настройки, без записи и загрузок.
make registries DRY_RUN=1

# Создать pip.conf, uv.toml и npmrc с корпоративными реестрами.
make registries

# Установить uv, окружение проекта и недостающий Node.js.
make bootstrap

# Проверить файлы и показать фактически используемые каталоги.
make config-validate
make config-show
make doctor
```

`DRY_RUN=1` показывает план, но не проверяет доступность сервисов и токены.
При первом bootstrap `uv.lock` может измениться: зависимости привязываются
к корпоративному индексу. Перед обновлением репозитория просмотрите изменения
и сохраните нужные, не отбрасывая их вслепую.

Созданные файлы находятся в `$SETUP_WORK_HOME/registries/`; переменные из `env.sh`
подключают их для pip, uv и npm. После смены токена повторите `make registries`.
После смены контейнера сначала проверьте системный Python, HDFS и Kerberos:
эти инструменты должны быть доступны в новом контейнере отдельно.

## 8. Установить модель и запустить API

Сначала посмотрите железо и оценку памяти:

```bash
make hardware
make recommend TASK=coding CONTEXT=8192 EXPLAIN=1
```

Для GGUF через llama.cpp:

```bash
make engine-install ENGINE=llama.cpp
make model-pull MODEL=qwen3.6-27b:q4_k_m
make serve MODEL=qwen3.6-27b:q4_k_m ENGINE=llama.cpp CONTEXT=8192
make serve-test
make serve-logs
make serve-stop
```

Для AWQ через vLLM:

```bash
make engine-install ENGINE=vllm
make model-pull MODEL=qwen3.8-27b:awq
make serve MODEL=qwen3.8-27b:awq ENGINE=vllm CONTEXT=8192
make serve-test
```

Для полной модели используйте `MODEL=qwen3.8-27b:full`.
Контекст `8192` — пример: допустимый запуск зависит от модели, железа и движка.
По умолчанию API доступен по `http://127.0.0.1:8000/v1`.

Агент устанавливается отдельно:

```bash
make harness-install NAME=qwen-code DRY_RUN=1
make harness-install NAME=qwen-code
make harness-doctor NAME=qwen-code
```

Адрес API и ключ для самого агента задаются по его настройкам. В `.env.example`
есть заготовки `OPENAI_BASE_URL`, `OPENAI_API_KEY` и `SETUP_WORK_API_KEY`.
Файл `.env` автоматически не читается: при использовании загрузите его явно
в shell. Саму команду агента запускайте после подключения `env.sh`.

## 9. Что делать в новом shell или контейнере

Откройте корень репозитория на постоянном томе и выполните:

```bash
. "$HOME/nfs/setup-work/env.sh"
export SETUP_WORK_USER_CONFIG_DIR="$PWD/config"
kinit
make doctor
```

Используйте ваш фактический путь к `env.sh`. Необходимость `kinit` зависит от
настроек кластера и срока действия билета. Если Python или путь к нему изменился,
может потребоваться `make sync` и повторная установка подходящего движка.

```text
$SETUP_WORK_HOME/
├── env.sh             подключение PATH и настроек инструментов
├── gitconfig          настройки Git
├── registries/        pip.conf, uv.toml, npmrc
├── bin/               ссылки на uv, Node.js и команды агентов
├── data/
│   ├── models/        распакованные модели
│   ├── engines/       распакованные движки
│   └── harnesses/     установленные агенты
├── state/             состояние серверов и логи
└── cache/             кэши uv и npm
```

## Если команда сообщает об ошибке

Сообщения самого setup-work и подсказки выводятся на русском. Текст внешних
инструментов, системных библиотек и серверов может оставаться на их языке.

| Ситуация | Что исправить |
|---|---|
| Не заполнен `CHANGE-ME` | Откройте названный файл и поле; параметры моделей находятся в конце `config/models.yaml` |
| Нет корпоративного PyPI | Заполните `SETUP_WORK_PYPI_INDEX` в `config/bootstrap.env`, выберите каталог через `SETUP_WORK_USER_CONFIG_DIR`, повторите `make registries` |
| Не читается файл токена | Создайте файл по пути `AUTH_FILE`, проверьте права либо очистите настройку для реестра без аутентификации |
| Ошибка сертификата | Укажите реальный корпоративный CA для pip/uv/npm/Git и проверьте доступность файла |
| Не найден uv или Node.js | Укажите источник uv в `bootstrap.env`, Node.js/npm — в `bootstrap.env` (npm registry), проверьте `PATH` после подключения `env.sh` |
| Не установлен Python нужной версии | Установите его в контейнере или подготовьте совместимый архив vLLM |
| Нет билета Kerberos | Выполните `kinit` и повторите команду |
| Нет файла на HDFS | Проверьте `hdfs dfs -ls` для конкретного `uri` из `corporate.yaml` |
| Не найден `llama-server` или `pyvenv.cfg` | Проверьте содержимое ZIP и `engines.<движок>.path` |
| Git не может прочитать репозиторий | Проверьте URL, `ref`, CA и credential helper через `git ls-remote` |
| Не хватает памяти | Уменьшите `CONTEXT`/`CONCURRENCY`, выберите меньший вариант; посмотрите `make recommend EXPLAIN=1` |
| Сервер долго не готов | Посмотрите `make serve-logs FOLLOW=1`; большие модели загружаются долго |
| Конфигурация читается из другого места | Проверьте `SETUP_WORK_USER_CONFIG_DIR` и вывод `make config-show` |
| Чужой `uv.toml` | Перенесите нужные настройки и сохраните резервную копию файла; затем освободите путь для `make registries` |

## Остальные команды и разработка

`make help` показывает все команды. `DRY_RUN=1` выводит план, `JSON=1` включает JSON
там, где он поддерживается, `YES=1` автоматически подтверждает запрошенные действия.

```bash
make model-list
make engine-status
make serve-status
make harness-list
make zip-join ZIP=/путь/к/разбитому/архиву
make check
```

`make check` выполняет ruff, pytest и проверку конфигурации. Тестам не нужны настоящие GPU, HDFS и доступ
в интернет, но серверным тестам требуется возможность открыть локальный порт.
Имена полей YAML, переменных окружения, команд и ключей JSON остаются на английском.

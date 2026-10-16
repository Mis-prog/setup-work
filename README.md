# setup-work

Настройка рабочего места для AI/ML во внутренней сети. В публичный интернет
(GitHub, pypi.org, nodejs.org, Hugging Face) не ходит.

- **pip, npm, uv** — корпоративные реестры в `pip.conf`, `.npmrc`, `uv.toml`;
- **uv и Node.js** — из корпоративных архивов;
- **модели и движки** (llama.cpp, vLLM) — zip с HDFS / NFS / внутреннего HTTPS;
- **подбор модели под железо** и запуск OpenAI-совместимого сервера;
- **харнессы** (qwen-code, opencode, pi, hermes, deepseek) и **skillhub** из Bitbucket.

## Быстрый старт

Нужны `git`, `make`, `python3` ≥ 3.11 (зеркала Python нет — он должен стоять на машине),
для HDFS — клиент `hdfs` и билет Kerberos (`kinit`).

```bash
git clone https://<bitbucket-host>/scm/<PROJECT>/setup-work.git && cd setup-work
make init          # создаст ~/.config/setup-work/ из шаблонов — заполните их
make bootstrap     # pip/npm/uv → uv → окружение проекта → Node.js
make doctor        # что настроено и чего не хватает
```

`~/.local/bin` должен быть в `PATH` (setup-work профиль shell не правит):
`export PATH="$HOME/.local/bin:$PATH"`.

## Где что указать

Все настройки — в одной папке **`~/.config/setup-work/`** (`make init` кладёт туда шаблоны):

| Файл | Что в нём |
|---|---|
| `bootstrap.env` | **URL PyPI и npm**, пути к файлам с токенами, откуда взять uv |
| `corporate.yaml` | пути к **моделям, движкам, архивам uv/Node.js** (hdfs://, /nfs/…, https://), хост Bitbucket |
| `harnesses.yaml` | **репозитории харнессов и skillhub** в Bitbucket (`repository`, `ref`) |

Пока в файле остаётся `CHANGE-ME`, соответствующая установка не запустится.
Токены в файлы не пишутся — только пути к файлам с ними.

### bootstrap.env → pip, npm, uv

```sh
SETUP_WORK_PYPI_INDEX=https://<repo>/<path>/pypi/simple
SETUP_WORK_PYPI_USER=token                 # если нужен логин
SETUP_WORK_PYPI_AUTH_FILE=~/nfs/token.txt  # файл с паролем/токеном
SETUP_WORK_PYPI_TRUSTED_HOST=<repo-host>   # только если ругается на сертификат
SETUP_WORK_NPM_REGISTRY=https://<repo>/<path>/npm/
SETUP_WORK_NPM_AUTH_FILE=~/nfs/tokens/npm-token-base64.txt
SETUP_WORK_UV_FROM_PIP=1                   # uv из корпоративного PyPI (или SETUP_WORK_UV_ARCHIVE=…)
```

`make registries` записывает (права 600):

| Файл | Что пишется |
|---|---|
| `~/.config/pip/pip.conf` | `index-url`, `trusted-host` (остальные ваши настройки сохраняются) |
| `~/.config/uv/uv.toml` | индекс по умолчанию, `python-downloads = "never"` (файл целиком ведёт setup-work) |
| `~/.npmrc` | `registry`, токен, `strict-ssl`/`cafile` (остальное сохраняется) |

Каждый инструмент читает свой файл сам, поэтому `pip`, `uv` и `npm` работают через
корпоративный реестр и вне setup-work. После смены токена — снова `make registries`.

## Команды

`make help` — полный список. Параметры передаются как переменные make.

```bash
# модели и движки
make recommend TASK=coding CONTEXT=32768 EXPLAIN=1     # что влезет в это железо и почему
make model-pull MODEL=qwen3-8b:q4_k_m                  # zip из corporate.yaml → распаковка
make zip-join ZIP=/data/Qwen3.8-Flash                  # склеить .zip + .z01 + .z02 … (с подтверждением)
make engine-install ENGINE=llama.cpp                   # или ENGINE=vllm (venv из zip, пути чинятся)

# сервер
make serve MODEL=qwen3-8b:q4_k_m CONTEXT=16384         # 127.0.0.1:8000, OpenAI API
make serve-test ; make serve-logs FOLLOW=1 ; make serve-stop

# харнессы и skillhub
make harness-install NAME=qwen-code
make harness-install NAME=skillhub
make harness-doctor
```

Общие переключатели: `DRY_RUN=1` — только показать план, `YES=1` — подтвердить всё,
`JSON=1` — вывод в JSON. Подбор под другую машину: `make hardware-json > hw.json` там,
`make recommend HARDWARE=hw.json` здесь (готовые профили — в `examples/hardware/`).

## Как упаковать артефакты

| Что | Содержимое zip | Ключ в `corporate.yaml` |
|---|---|---|
| GGUF для llama.cpp | `.gguf` (шарды `-00001-of-0000N`), можно в подкаталоге | `models: "<id>:<variant>"` |
| Веса для vLLM | каталог HF: `config.json`, `*.safetensors`, токенизатор | `models: "<id>:<variant>"` |
| llama.cpp | собранный `llama-server` и его `.so` рядом | `engines: llama.cpp` |
| vLLM | каталог venv, упакованный `zip -r` | `engines: vllm` |
| uv, Node.js | официальные архивы (sha256 проверяется по `config/defaults.yaml`) | `tools: uv / node` |

`id:variant` — из `make models`. Свои модели — в `~/.config/setup-work/models.yaml`
(пример: `examples/corp/models.yaml`). venv vLLM собирайте на той же ОС, версии Python и
CUDA, что у целевых машин: после распаковки setup-work перепривязывает его к
установленному Python той же версии и переписывает shebang в `bin/*`.

Повторный `model-pull` ничего не качает, если URI и sha256 не изменились. Нужен запас
места ~2× архива (`SETUP_WORK_MODEL_DIR` — куда распаковывать).

## Харнессы из Bitbucket

Ссылка — из кнопки Clone: `https://<bitbucket-host>/scm/<PROJECT>/<repo>.git`. Логин и
токен git берёт из credential helper (`git config --global credential.helper store`),
корпоративный CA — `git config --global http.sslCAInfo /path/to/ca.crt`. Хост добавьте в
`trusted_git_hosts` в `corporate.yaml`, тогда команды сборки из рецепта не требуют
`ALLOW_BUILD=1`. Харнесс ставится в `~/.local/share/setup-work/harnesses/<name>/`,
бинарники линкуются в `~/.local/bin`.

## Решение проблем

| Симптом | Решение |
|---|---|
| `no corporate PyPI configured` | заполните `SETUP_WORK_PYPI_INDEX` в `bootstrap.env`, затем `make registries` |
| `Failed to fetch https://files.pythonhosted.org/…`, `dns error` | uv не видит корпоративный индекс: `make registries`, проверьте `~/.config/uv/uv.toml` |
| `uv.toml exists and was not written by setup-work` | перенесите свои настройки и удалите файл, затем `make registries` |
| `uv.lock` изменён после `make` | так и должно быть (lock пересобран под корпоративный индекс); перед `git pull` — `git checkout uv.lock` |
| `no corporate source for uv` / `for node` | `SETUP_WORK_UV_FROM_PIP=1` или `SETUP_WORK_UV_ARCHIVE` в `bootstrap.env`; `tools.node` в `corporate.yaml` |
| `Python … is not installed` | Python не скачивается: поставьте нужную версию на машину (пакет, админы) |
| `no valid Kerberos ticket` | `kinit`, затем повторить |
| `does not exist on HDFS` | проверьте путь: `hdfs dfs -ls <uri>` |
| `has no artifact source` | добавьте `"<id>:<variant>"` в `models:` в `corporate.yaml` |
| `estimated not to fit` | меньше `CONTEXT`/`CONCURRENCY`, вариант поменьше, `TP=`, или `FORCE=1` |
| `still contains the CHANGE-ME placeholder` | заполните `repository`/`ref` в `harnesses.yaml` |
| `server did not become ready` | большие модели грузятся минутами: `make serve-logs` |

## Безопасность

- Никаких публичных хостов, `curl | sh`, sudo и правок профиля shell.
- Подпроцессы — только списком аргументов; переменные make идут через окружение.
- Токены читаются из файлов, не печатаются и не попадают в argv; в выводе маскируются.
- Удаление (`model-remove`, `harness-remove`, `clean`) показывает, что будет удалено, и
  спрашивает подтверждение.
- Сервер слушает `127.0.0.1`; другой адрес — только с подтверждением, ключ API
  передаётся через файл или окружение.

## Разработка

```bash
make check     # ruff + pytest + проверка конфигурации
make format
```

Вне корпоративной сети задайте индекс явно: `export UV_DEFAULT_INDEX=<url>/simple`.
Тесты не требуют GPU, сети и HDFS (всё замокано, профили железа — в `examples/hardware/`).

```text
Makefile ─ SETUP_WORK_* ─▶ setup_work.cli
  scripts/bootstrap.sh  pip.conf / uv.toml / .npmrc, uv, uv sync   (до Python-окружения)
  config.py      YAML: config/ + ~/.config/setup-work/
  bootstrap.py   uv, Node.js, движки из corporate.yaml
  hub.py, artifacts.py   модели: HDFS/NFS/HTTPS → распаковка
  hardware.py, memory.py, models.py, engines.py   железо → оценка памяти → подбор
  inference.py   запуск и остановка сервера
  harness.py     харнессы и skillhub
  steps.py, runner.py    план → выполнение (dry-run, подтверждения, маскирование)
```

# setup-work во внутренней корпоративной среде

Шаблон командного конфига: веса моделей, llama.cpp и venv vLLM лежат zip-архивами на
HDFS, PyPI и npm настраиваются корпоративными скриптами, харнесы собираются из
внутренних git-форков. Задача: коллега выполняет пару команд и получает рабочее
окружение без ручных шагов и без интернета.

> Статус: весь поток (HDFS → zip → распаковка → `serve`, venv из zip, внутренние
> харнесы) покрыт тестами с подставным `hdfs` CLI. На настоящем кластере с Kerberos
> его **ещё не прогоняли**. Первый запуск у вас — проверка, сообщайте, что не так.

## 1. Один раз для команды: подготовить конфиг

1. Скопируйте `examples/corp/` в отдельный репозиторий (например
   `git.corp/ai-tools/setup-work-team`).
2. Заполните:
   - `corporate.yaml`: пути на HDFS к весам, llama.cpp и venv vLLM; пути к
     корпоративным скриптам PyPI/npm; `trusted_git_hosts`;
   - `harnesses.yaml`: реальные `repository` и `ref` (тег или commit) для qwen-code,
     opencode, pi, hermes, deepseek. Проверьте команды `build` и пути `binaries`
     под ваши форки;
   - `bootstrap.env`: адреса корпоративных PyPI и npm и пути к файлам с токенами
     (`SETUP_WORK_PYPI_*`, `SETUP_WORK_NPM_*`), а также как получить uv без GitHub
     (корпоративный PyPI или HDFS);
   - `models.yaml`: внутренние fine-tune модели, если они есть.
3. Проверьте конфиг:
   ```bash
   SETUP_WORK_TEAM_CONFIG_DIR=$PWD/setup-work-team make config-validate
   ```
   Пока где-то остаётся `CHANGE-ME`, установка такого харнеса откажется работать.

### Как упаковывать артефакты

| Что | Содержимое zip | Ключ в `corporate.yaml` |
|---|---|---|
| GGUF для llama.cpp | один или несколько `.gguf` (шарды `-00001-of-0000N`), можно в подкаталоге | `models: "<id>:<variant>"` |
| Веса для vLLM | каталог в формате HF: `config.json`, `*.safetensors`, токенизатор | `models: "<id>:<variant>"` |
| llama.cpp | собранный `llama-server` и его `.so` рядом | `engines: llama.cpp` (`path:` — если бинарник не один) |
| vLLM | каталог venv, упакованный `zip -r` | `engines: vllm` (`path:` — каталог venv в архиве) |

Желательно указывать `sha256` (`sha256sum file.zip`): тогда setup-work проверит копию.

**venv для vLLM.** Обычный `zip -r` venv'а не переносится как есть: в `pyvenv.cfg` и
`bin/*` зашиты пути машины, где его собирали. setup-work после распаковки:
- проверяет, запускается ли `bin/python`;
- если нет, ищет Python той же версии (`uv python find`, `pythonX.Y`) или предлагает
  `uv python install X.Y`, затем перепрописывает `home` в `pyvenv.cfg` и ссылки
  `bin/python*`;
- переписывает shebang в `bin/*`;
- запускает vLLM как `bin/python -c "…vllm main…" serve …`, поэтому даже битые
  shebang ему не мешают.

Собирайте venv на той же ОС, с той же версией Python и CUDA, что у целевых машин.
Удобнее всего — Python от uv (`uv venv --python 3.12`).

## 2. У каждого коллеги

```bash
kinit user@CORP.REALM                     # билет Kerberos для HDFS (пароль вводите сами)
git clone <setup-work> && cd setup-work
git clone <team-config> ~/setup-work-team
export SETUP_WORK_TEAM_CONFIG_DIR=~/setup-work-team
export PATH="$HOME/.local/bin:$PATH"      # добавьте в ~/.bashrc сами — setup-work их не трогает

make bootstrap DRY_RUN=1                  # посмотреть план
make bootstrap                            # корп. скрипты → uv → venv проекта → Node/git-lfs
make doctor                               # в разделе «Corporate»: HDFS, Kerberos, индексы
make setup PROFILE=corp-dev               # внутренние харнесы
```

На GPU-сервере:

```bash
make setup PROFILE=corp-gpu DRY_RUN=1
make engine-install ENGINE=vllm           # venv из HDFS, с починкой путей
make engine-install ENGINE=llama.cpp      # сборка из HDFS
make recommend TASK=coding CONTEXT=32768
make model-pull MODEL=qwen3-coder-30b-a3b:fp8   # zip из HDFS → распаковка → проверка
make serve MODEL=qwen3-coder-30b-a3b:fp8 ENGINE=vllm CONTEXT=32768
make serve-test
```

Харнесы к локальному серверу:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=local
qwen    # / opencode / pi / hermes / deepseek
```

## Что происходит под капотом

| Шаг | Поведение |
|---|---|
| `make bootstrap` (stage 0, без Python) | читает `$SETUP_WORK_TEAM_CONFIG_DIR/bootstrap.env` (только `KEY=VALUE`, секреты отклоняются), запускает `SETUP_WORK_PRE_SCRIPTS` с подтверждением, пишет `pip.conf` и `.npmrc` из `SETUP_WORK_PYPI_*`/`SETUP_WORK_NPM_*` (токены из файлов, права 600, прочие настройки сохраняются), берёт uv из HDFS/PyPI/GitHub, передаёт `index-url` из pip.conf в uv |
| uv и `uv.lock` | `uv.lock` содержит ссылки на pypi.org; при корпоративном индексе lock пересобирается против него (версии из lock сохраняются как предпочтительные) |
| Корпоративные скрипты | `make corp-setup`: каждый показывается и подтверждается; пропускается, если файлы из `creates` уже есть |
| pip.conf → uv | uv не читает pip.conf: `index-url`, `extra-index-url`, `trusted-host` и `cert` превращаются в `UV_*` и `SSL_CERT_FILE` для всех вызовов uv |
| HDFS | `hdfs dfs -du -s` (размер, проверка места) и `hdfs dfs -get` во временный `.part`, затем атомарное переименование. Без валидного билета (`klist -s`) работа останавливается с подсказкой `kinit` |
| Распаковка | проверки path traversal; права и симлинки сохраняются; распаковка во временный каталог и атомарная замена; zip после успешной распаковки удаляется (`keep_archive: true` — оставить) |
| Повторный запуск | маркер `.setup-work-artifact.json`: тот же URI и sha256 — ничего не качается |
| Харнесы из git | клон своей git-аутентификацией (`GIT_TERMINAL_PROMPT=0`), checkout `ref`, шаги `build` (для `trusted_git_hosts` без `ALLOW_BUILD=1`), обёртки `bin/<name>` → `node/bun/python <файл>`, симлинк в `~/.local/bin` |

## Диагностика

```bash
make corp-status          # HDFS CLI, Kerberos, pip/uv/npm индексы, сколько артефактов
make corp-env             # export-строки корпоративного окружения (для CI/шелла)
make model-list           # распакованные модели
make harness-doctor NAME=qwen-code
```

| Симптом | Решение |
|---|---|
| `no valid Kerberos ticket` | `kinit`, затем повторить команду |
| `does not exist on HDFS` | проверьте путь: `hdfs dfs -ls <uri>` |
| `not enough disk space` | нужно ~2× размера архива; задайте `SETUP_WORK_MODEL_DIR` или `XDG_DATA_HOME` на большой том |
| `the vLLM venv was built with Python 3.X … not available` | `uv python install 3.X` (с `UV_PYTHON_INSTALL_MIRROR`) или соберите venv на той же версии |
| `uv sync failed` | индекс: `make corp-env`, `UV_DEFAULT_INDEX`, pip.conf; прокси |
| `cannot read … auth file` | проверьте путь к файлу с токеном в `bootstrap.env` (`SETUP_WORK_PYPI_AUTH_FILE`, `SETUP_WORK_NPM_AUTH_FILE`) |
| сменился токен | `make corp-registries` перезапишет `pip.conf` и `.npmrc` |
| харнес: `still contains the CHANGE-ME placeholder` | заполните `repository`/`ref` в командном `harnesses.yaml` |

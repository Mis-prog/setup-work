# Пример HDFS-настроек для трёх моделей Qwen 27B

Это заготовки для ручной правки. Адреса архивов примерные, параметры моделей
не проверены. Рабочие настройки и встроенный каталог проекта не изменены.

| Ваш архив | Идентификатор в командах | Движок |
|---|---|---|
| `qwen3.8-27b.zip` | `qwen3.8-27b:full` | vLLM |
| `qwen3.8-27b-awq.zip` | `qwen3.8-27b:awq` | vLLM |
| `qwen3.6-27b-gguf-Q4_K_M.zip` | `qwen3.6-27b:q4_k_m` | llama.cpp |

Для первых двух предполагаются safetensors. `full` предполагает неквантованные
веса: уточните это по вашим файлам. AWQ объединён с полной моделью как вариант
одной архитектуры; если это другая базовая модель, нужна отдельная запись.
Названия qwen3.8/qwen3.6 оставлены как указаны, без подмены другой версией Qwen.

## Что поправить

В `corporate.yaml` замените HDFS-пути к трём моделям и двум движкам.
`engines.llama.cpp.path` указывает на файл `llama-server` внутри ZIP;
`engines.vllm.path` — на каталог venv с `pyvenv.cfg`. Исходники llama.cpp
могут оставаться в архиве вместе с готовым `build/`; установщик их не собирает.
Если есть SHA256 архивов, добавьте `sha256` в соответствующие записи.

В `models.yaml` замените все активные `CHANGE-ME`:

- `architecture`: `dense` либо `moe`, согласно реальной модели;
- `context_length`: поддерживаемый моделью контекст;
- `attention`: параметры из `config.json`, при наличии — из его `text_config`,
  либо метаданных GGUF. `num_layers` соответствует `num_hidden_layers`,
  `num_kv_heads` — `num_key_value_heads`;
- `dtype`: тип весов полной модели и тип активаций AWQ;
- `size_gb`: фактический размер весов в десятичных GB;
- `license`: лицензия вашей модели. Уточните также номинальные `parameters_b`.

Для MoE дополнительно заполните `active_parameters_b` и `moe`.
Текущая схема проекта описывает обычное внимание и MoE. Если в реальном
`config.json` есть гибридные типы слоёв или иные механизмы внимания, оценка памяти
может потребовать доработки кода: одной замены чисел недостаточно.
Проверка YAML не доказывает поддержку архитектуры установленными движками.

## Как применить

Сначала отредактируйте примеры. Из корня репозитория, после `make init` и
подключения `env.sh`, скопируйте оба файла:

```bash
cp -i examples/corp/qwen-27b/corporate.yaml "$SETUP_WORK_HOME/config/corporate.yaml"
cp -i examples/corp/qwen-27b/models.yaml "$SETUP_WORK_HOME/config/models.yaml"
```

Если в целевых файлах уже есть ваши настройки, перенесите нужные записи вручную:
копирование целого файла заменяет его содержимое. Без `SETUP_WORK_HOME` стандартный
каталог — `~/.config/setup-work/` (если не заданы XDG или другие переопределения).
`bootstrap.env` для PyPI/npm/первой установки uv настраивается отдельно.
Если при bootstrap требуется Node.js и его ещё нет, заполните закомментированный
раздел `tools.node` в `corporate.yaml`.

До замены плейсхолдеров `make config-validate` выдаёт ошибки — это ожидаемо.
После заполнения:

```bash
make config-validate
make models
kinit
```

Для GGUF:

```bash
make engine-install ENGINE=llama.cpp
make model-pull MODEL=qwen3.6-27b:q4_k_m
make serve MODEL=qwen3.6-27b:q4_k_m ENGINE=llama.cpp CONTEXT=8192
make serve-test
make serve-stop
```

Для AWQ:

```bash
make engine-install ENGINE=vllm
make model-pull MODEL=qwen3.8-27b:awq
make serve MODEL=qwen3.8-27b:awq ENGINE=vllm CONTEXT=8192
make serve-test
```

Для полной модели замените `qwen3.8-27b:awq` на `qwen3.8-27b:full`.
Контекст `8192` в командах — пример, запуск также зависит от памяти и поддержки
модели движком. Пользовательский каталог добавляет эти модели к встроенным;
встроенные записи не удаляются.

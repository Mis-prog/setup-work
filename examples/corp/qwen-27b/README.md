# Проверенные параметры трёх моделей Qwen 27B

Параметры заполнены также прямо в `config/models.yaml` репозитория.
Пути HDFS в `corporate.yaml` и `source` остаются примерами: замените их адресами
ваших архивов. Архивы, сертификаты, токены и сборки движков не проверялись.

| Идентификатор | Веса | Размер файлов весов, GB | Движок |
|---|---|---:|---|
| `qwen3.8-27b:full` | BF16 safetensors | 55.563006776 | vLLM |
| `qwen3.8-27b:awq` | AWQ INT4, активации BF16 | 21.018000928 | vLLM |
| `qwen3.6-27b:q4_k_m` | GGUF Q4_K_M | 16.817244384 | llama.cpp |

Размеры указаны в десятичных GB: байты / 1 000 000 000. Для safetensors это сумма
размеров файлов, включая заголовки, а для GGUF — размер выбранного файла.
Полный вариант содержит vision и MTP; параметров 27.781427952B.
Текстовая GGUF-модель по метаданным репозитория содержит 26.895998464B параметров.
Лицензия всех трёх источников — Apache-2.0.

## Источники

Данные прочитаны 2026-10-19, без скачивания весов:

- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B):
  ревизия `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
  [config.json](https://huggingface.co/Qwen/Qwen3.8-27B/raw/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json).
  Размеры 18 файлов safetensors и число параметров взяты из API репозитория.
- [cyankiwi/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4):
  ревизия `6e134bae811fb5adac50ee042ae5f029ac6779aa`;
  [config.json](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4/raw/6e134bae811fb5adac50ee042ae5f029ac6779aa/config.json).
  Размеры 5 файлов safetensors взяты из API репозитория.
- [Qwen3.6-27B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/blob/82d411acf4a06cfb8d9b073a5211bf410bfc29bf/Qwen3.6-27B-Q4_K_M.gguf):
  размер `16817244384` байт, ревизия `82d411acf4a06cfb8d9b073a5211bf410bfc29bf`.
  Источник архитектуры — [Qwen/Qwen3.6-27B/config.json](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json),
  указанная базовая модель конвертации Unsloth.

Веса vLLM берутся из папок с частями `<имя>-1-of-N.zip`;
для них в corporate.yaml указано `format: zip-parts`.

## Архитектура и ограничения оценки памяти

Обе модели — плотные, с нативным контекстом 262144 токена и гибридным вниманием:
64 слоя, из них 48 Gated DeltaNet и 16 полного внимания. У полного внимания
24 Q-головы, 4 KV-головы, размер головы 256.

В `attention.num_layers` сохранено реальное значение 64. Текущая формула setup-work
считает все слои обычными KV-слоями и отдельно не моделирует состояние DeltaNet.
Поэтому оценка `recommend` и проверка памяти при `serve` для этих моделей
приближённые; поддержку гибридной архитектуры нужно проверить на вашей сборке движка.

AWQ-репозиторий использует `quant_method: compressed-tensors`, а не старый
контейнер AutoAWQ. Сохраните его исходный `config.json` вместе с весами.
Запуск vLLM определяет фактический метод квантования из этого файла.
Не заменяйте его на конфигурацию полной модели. Часть тензоров остаётся BF16,
поэтому размер AWQ не равен четверти размера полной модели.

## Использование

Для работы прямо с настройками репозитория из его корня:

```bash
export SETUP_WORK_USER_CONFIG_DIR="$PWD/config"
make config-validate
make model-pull MODEL=qwen3.6-27b:q4_k_m
make engine-install ENGINE=llama.cpp
make serve MODEL=qwen3.6-27b:q4_k_m ENGINE=llama.cpp CONTEXT=8192
```

Для AWQ используются `MODEL=qwen3.8-27b:awq` и `ENGINE=vllm`,
для полной модели — `MODEL=qwen3.8-27b:full` и `ENGINE=vllm`.
Сначала подключите ваше `env.sh`, настройте реестры и доступ к HDFS.
Для прямого редактирования основных файлов копировать эти примеры не требуется.

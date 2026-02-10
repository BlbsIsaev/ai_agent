# Browser Agent

Автономный AI-агент для управления браузером и выполнения многошаговых задач. Архитектура: DOM-first с vision fallback, контекстное планирование и security layer для рискованных действий. Поддерживаются persistent sessions и видимый браузер.

## Возможности
- Управление браузером (видимый режим, persistent sessions).
- Инструменты: observe / click / type / find_elements / extract_* / vision_find / click_at / web_search.
- Обработка ошибок и восстановление при залипании.
- Security layer для подтверждения рискованных действий.

## Требования
- Python 3.11+
- Установленный Chromium/Chrome (или Playwright, если нужен bundled browser)

## Установка
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m playwright install
```

## Настройка
Создайте `.env` в корне (пример):
```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_CHEAP_MODEL=gpt-4.1-nano
OPENAI_VISION_MODEL=gpt-4.1-mini
USER_DATA_DIR=.user_data
SLOW_MO_MS=120
ENABLE_HANDOFF=1
ENABLE_WEB_SEARCH=1
AUTO_VISION_ON_STUCK=1
OBSERVE_MAX_BLIND_STEPS=3
CHROME_EXECUTABLE_PATH=/Applications/Google Chrome.app/Contents/MacOS/Google Chrome
```

## Запуск
```bash
python -m src.agent.main
```


# src/agent/orchestrator.py
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from collections import deque
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Optional
import string

from rich.console import Console

from .llm import OpenAIReActClient
from .memory import Memory
from .security import is_risky, ask_user_confirmation
from .tools_browser import BrowserController, Element
from .tools_perception import perceive_page, normalize_elements, get_visible_text, collect_raw_elements, collect_links

console = Console()

# ============================================================
# Readiness / anti-loop (SPA stabilization)
# ============================================================

_NOT_READY_PATTERNS = [
    r"enable javascript",
    r"javascript.*(disabled|required|must be enabled)",
    r"включите javascript",
    r"java ?script.*(нужен|требуется|должен быть включен)",
    r"\bloading\b",
    r"загрузка",
    r"подождите",
    r"please wait",
]

def snapshot_fingerprint(obs: dict) -> str:
    """Cheap fingerprint to detect 'same snapshot' loops."""
    url = (obs.get("url") or "")[:200]
    title = (obs.get("title") or "")[:200]
    txt = (obs.get("visible_text") or "")[:500]
    n = len(obs.get("elements") or [])
    payload = f"{url}\n{title}\n{n}\n{txt}".encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()


class LoopGuard:
    """
    Detect repeated identical snapshots (agent stuck / page not changing).
    """
    def __init__(self, window: int = 6, same_threshold: int = 3):
        self._buf = deque(maxlen=window)
        self.same_threshold = same_threshold

    def push(self, fp: str) -> Dict[str, Any]:
        self._buf.append(fp)
        streak = 1
        for i in range(len(self._buf) - 2, -1, -1):
            if self._buf[i] == fp:
                streak += 1
            else:
                break
        return {"streak": streak, "stuck": streak >= self.same_threshold}


def is_observation_not_ready(obs: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Heuristics: early snapshot / SPA not hydrated.
    Returns (not_ready, reason).
    """
    txt = (obs.get("visible_text") or "").lower()
    n = len(obs.get("elements") or [])
    has_rows = any(e.get("role") == "row" for e in (obs.get("elements") or []))
    has_search = any(
        (e.get("tag") in ("input", "textarea") and ("search" in (e.get("name") or "").lower() or "поиск" in (e.get("name") or "").lower()))
        for e in (obs.get("elements") or [])
    )

    for pat in _NOT_READY_PATTERNS:
        if re.search(pat, txt):
            # If we already have a populated UI (rows/search), don't treat as not-ready
            if n >= 12 and (has_rows or has_search):
                return False, "ok_ui_despite_marker"
            return True, f"marker:{pat}"

    # Too few interactive elements often indicates skeleton/fallback.
    if n < 12:
        return True, f"too_few_elements:{n}"

    if len(txt.strip()) < 120:
        return True, "too_little_text"

    return False, "ok"


async def wait_until_ready(
    observe_coro,
    wait_coro,
    *,
    max_attempts: int = 6,
    base_ms: int = 650,
    factor: float = 1.7,
    max_ms: int = 4000,
) -> Dict[str, Any]:
    """
    observe_coro: async fn() -> obs
    wait_coro: async fn(ms:int) -> None
    """
    delay = base_ms
    last: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_attempts + 1):
        obs = await observe_coro()
        last = obs

        not_ready, reason = is_observation_not_ready(obs)
        if not not_ready:
            obs["readiness"] = {"ready": True, "attempt": attempt, "reason": "ok"}
            return obs

        obs["readiness"] = {"ready": False, "attempt": attempt, "reason": reason, "next_wait_ms": delay}
        await wait_coro(delay)
        delay = min(int(delay * factor), max_ms)

    if last is None:
        return {"readiness": {"ready": False, "reason": "no_observation"}}
    last["readiness"] = {"ready": False, "reason": "max_attempts_exceeded"}
    return last


class StableObserver:
    """
    Wraps raw _observe() with SPA readiness waiting + loop detection.
    """
    def __init__(self):
        self.guard = LoopGuard(window=6, same_threshold=3)

    async def observe_stable(self, raw_observe, browser_wait) -> Dict[str, Any]:
        obs = await wait_until_ready(raw_observe, browser_wait)
        fp = snapshot_fingerprint(obs)
        lg = self.guard.push(fp)
        obs["loop_guard"] = lg

        if lg["stuck"]:
            if lg["streak"] <= self.guard.same_threshold:
                obs["stuck_hint"] = (
                    "Страница не меняется (одинаковый снапшот). "
                    "Попробуй один раз: wait(1500), press('Escape'), затем observe() снова."
                )
            else:
                obs["stuck_hint"] = (
                    "Страница всё ещё не меняется. Не скролль дальше. "
                    "Лучше попробуй: использовать поиск на странице (если есть), "
                    "click_best_match() по ключевому запросу, back(), либо reload()."
                )
        return obs


# ============================================================
# Tools schema
# ============================================================

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "observe",
        "description": "Снять текущее состояние страницы (url, title, краткий видимый текст, интерактивные элементы).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "extract_text",
        "description": "Получить расширенный видимый текст страницы (для чтения контента).",
        "parameters": {
            "type": "object",
            "properties": {"max_chars": {"type": "integer", "description": "Максимум символов, по умолчанию 4000."}},
        },
    },
    {
        "type": "function",
        "name": "extract_links",
        "description": "Извлечь список видимых ссылок (текст + href).",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Сколько ссылок вернуть, по умолчанию 60."}},
        },
    },
    {
        "type": "function",
        "name": "extract_list",
        "description": "Структурированный список элементов (строки/карточки) по роли/тегу.",
        "parameters": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "ARIA role (row, listitem, option, gridcell и т.п.)"},
                "tag": {"type": "string", "description": "HTML тег (tr, li, div и т.п.)"},
                "limit": {"type": "integer", "description": "Сколько элементов вернуть, по умолчанию 30."},
            },
        },
    },
    {
        "type": "function",
        "name": "goto",
        "description": "Перейти по URL.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    },
    {
        "type": "function",
        "name": "click",
        "description": "Клик по element_id из observe(). Если совпадений несколько — index (nth).",
        "parameters": {
            "type": "object",
            "properties": {
                "element_id": {"type": "string"},
                "index": {"type": "integer", "description": "nth(index) если совпадений несколько. По умолчанию 0."},
            },
            "required": ["element_id"],
        },
    },
    {
        "type": "function",
        "name": "type",
        "description": "Ввести текст в поле по element_id. Если совпадений несколько — index (nth). Для поиска можно submit=true, чтобы сразу нажать Enter.",
        "parameters": {
            "type": "object",
            "properties": {
                "element_id": {"type": "string"},
                "text": {"type": "string"},
                "clear": {"type": "boolean"},
                "submit": {"type": "boolean", "description": "После ввода нажать Enter (полезно для поиска)."},
                "index": {"type": "integer", "description": "nth(index) если совпадений несколько. По умолчанию 0."},
            },
            "required": ["element_id", "text"],
        },
    },
    {
        "type": "function",
        "name": "press",
        "description": "Нажать клавишу (Enter, Escape, Ctrl+L и т.п.).",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
    },
    {
        "type": "function",
        "name": "reload",
        "description": "Перезагрузить текущую страницу.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "click_at",
        "description": "Клик по координатам (x,y). Используй, если элемент не находится в DOM.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
    },
    {
        "type": "function",
        "name": "scroll",
        "description": "Прокрутка (dy>0 вниз, dy<0 вверх).",
        "parameters": {"type": "object", "properties": {"dy": {"type": "integer"}}},
    },
    {"type": "function", "name": "back", "description": "Назад.", "parameters": {"type": "object", "properties": {}}},
    {
        "type": "function",
        "name": "wait",
        "description": "Подождать N миллисекунд.",
        "parameters": {"type": "object", "properties": {"ms": {"type": "integer"}}},
    },
    {
        "type": "function",
        "name": "click_best_match",
        "description": "Найти и кликнуть лучший элемент по описанию (универсальная дизамбигуация).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Альтернативные формулировки запроса (синонимы/локализация).",
                },
                "prefer": {"type": "string", "enum": ["topmost", "largest", "default"]},
                "role_hint": {"type": "string"},
                "vision_fallback": {"type": "boolean", "description": "Разрешить fallback на vision_find + click_at."},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "find_elements",
        "description": "Найти элементы по запросу/ролям и вернуть список кандидатов с element_id для клика/ввода.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Текстовый запрос (можно пустой для списка по роли)."},
                "role": {"type": "string", "description": "ARIA role (row, listitem, button, link и т.п.)"},
                "tag": {"type": "string", "description": "HTML тег (a, button, input и т.п.)"},
                "limit": {"type": "integer", "description": "Сколько кандидатов вернуть, по умолчанию 20."},
            },
        },
    },
    {
        "type": "function",
        "name": "vision_find",
        "description": "Найти координаты элемента по описанию на скриншоте (визуальный поиск).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

WEB_SEARCH_TOOL = {"type": "web_search"}


# ============================================================
# System prompts
# ============================================================

SYSTEM_BASE = r"""
Ты — автономный AI-агент, управляющий веб-браузером через tools. Твоя цель — выполнить задачу пользователя в реальном сайте.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА
1) Работай итеративно: OBSERVE → PLAN → ACT(1-3 действия) → VERIFY(OBSERVE по триггерам).
2) Не выдумывай состояние страницы. Если чего-то не знаешь — сделай observe().
   Если observe показывает пустую страницу (about:blank или нет элементов/текста) — не ищи кнопки, сразу делай goto на сайт.
   Если адрес неизвестен — используй web_search для нахождения официального домена, затем goto.
3) Нельзя хардкодить селекторы, URL-пути страниц, названия кнопок для конкретного сайта. Всё находи из observe().
4) Всегда кликай/вводи по element_id из observe(). Если элементов много и совпадения неоднозначны — используй click_best_match(query=...).
   Если нужен клик по названию, используй click_best_match с queries=[...] (синонимы) — при необходимости он сделает vision fallback.
5) Держи расходы токенов низкими: не пиши длинные объяснения. Observe вызывай по триггерам:
   - после goto/back/reload
   - после submit/Enter в поиске/форме
   - после ошибок/таймаутов
   - если не уверен, что достиг цели
6) Если нужен логин/2FA/капча — попроси пользователя сделать это вручную. Затем продолжай.
   Если пользователь просит действие на сайте, предполагается, что он уже залогинен там (если не видно обратного).
7) Поиск: после ввода в поле поиска обязательно инициируй поиск — нажми Enter или кликни кнопку поиска/лупу, затем observe().
   Если уместно, используй type(..., submit=true), чтобы после ввода сразу нажать Enter.
8) Для чтения длинного контента используй extract_text(max_chars=...), а для поиска нужных элементов вне observe() — find_elements().
9) Для списков/карточек используй extract_list(role=.../tag=...), а для ссылок — extract_links().
10) Если элемент невозможно найти в DOM, используй vision_find(query=...) и затем click_at(x,y).
11) Если click() не сработал (таймаут/блокировка), попробуй click_at по центру элемента.
12) Если после клика страница не изменилась, попробуй альтернативный элемент или, если есть href, перейди по нему.
13) Если видишь сообщение о выключенном JavaScript или страница «зависла», используй reload().
14) Если задача требует N элементов (например “последние 3 письма”), обязательно пройди все N: после анализа одного элемента вернись к списку и продолжай, пока не достигнешь N.
15) Финальный ответ должен быть полным и завершённым; не обрывай предложения. Если не успеваешь — сократи детали, но закончи ответ.
16) Не завершай работу, пока цель не достигнута. Завершение возможно только если задача выполнена или требуется ввод/подтверждение пользователя.
17) Если в observe() есть vision_hint — используй его: попробуй click_at по координатам или сопоставь query с DOM через find_elements/click_best_match.

НЕ СПРАШИВАЙ БЕЗ НУЖДЫ
- Если задача ясна — действуй сразу. Не проси уточнений и не «готовься», а выполняй.
- Вопросы допустимы только когда реально не хватает данных (например, нужен логин/2FA/капча или пользователь должен подтвердить рискованное действие через security layer).
- Не заканчивай ответ фразой «Готово…» пока задача не выполнена.

SECURITY (рискованные / account actions)
Удаление/перемещение писем, пометка спама, отправка отклика/письма, оформление/подтверждение заказа, лайк/подписка — считаются account actions.
Не спрашивай подтверждение текстом. Если нужно рискованное действие — просто вызывай соответствующий tool, а security layer сам запросит подтверждение у пользователя.

ДИЗАМБИГУАЦИЯ
Если нужных элементов несколько — используй click_best_match() с уточняющим query. После клика делай observe() и проверяй результат.

САМОСТОЯТЕЛЬНОЕ ПЛАНИРОВАНИЕ
Твоя задача — самостоятельно составлять алгоритм действий в моменте, исходя из наблюдения страницы и цели пользователя.
Не используй заранее заданные сценарии для конкретных доменов или типов задач. Каждый шаг выбирай на основе текущего observe().
План может и должен меняться после каждого observe(), если страница изменилась или действие не сработало.

ФОРМАТ
- В начале ответа: "PLAN:" (1–3 кратких шага).
- Перед каждым действием: "ACTION:" (что именно собираешься сделать).
- После действия: "EXPECTATION:" (что ожидаешь увидеть после observe()).
- В конце: "SUMMARY:" (короткий список фактически выполненных результатов). Не описывай планы в SUMMARY.
Между шагами: 1–2 строки, без длинных рассуждений.
"""

def pick_instructions() -> str:
    return SYSTEM_BASE


# ============================================================
# click_best_match helpers
# ============================================================

_STOP = {
    "и", "в", "на", "из", "по", "для", "к", "у", "о", "об", "это", "что", "как",
    "the", "a", "an", "of", "to", "for", "and", "or",
}

def _tokenize(s: str) -> set[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-zа-я0-9]+", " ", s, flags=re.IGNORECASE)
    return {t for t in s.split() if t and t not in _STOP and len(t) > 1}

def _text_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _score_element(el: dict, query: str, role_hint: Optional[str], prefer: str) -> float:
    if el.get("disabled"):
        return -999.0

    q = (query or "").strip().lower()
    name = (el.get("name") or "")
    text = (el.get("text") or "")
    href = (el.get("href") or "")
    aria = (el.get("aria") or "")
    title = (el.get("title") or el.get("title_attr") or "")
    tooltip = (el.get("tooltip") or "")

    role = (el.get("role") or "")
    tag = (el.get("tag") or "")

    s = max(
        _text_sim(name, q),
        _text_sim(text, q),
        _text_sim(aria, q),
        _text_sim(title, q),
        _text_sim(tooltip, q),
    )

    tq = _tokenize(q)
    te = _tokenize(" ".join([name, text, aria, title, tooltip]))
    if tq:
        overlap = len(tq & te) / max(1, len(tq))
        s += 0.6 * overlap

    if role_hint:
        rh = role_hint.lower()
        if rh in role.lower() or rh == tag.lower():
            s += 0.25

    if tag == "a" or href:
        s += 0.15
    if tag == "button" or role == "button":
        s += 0.15

    bbox = el.get("bbox") or [0, 0, 0, 0]
    try:
        _, y, w, h = bbox
        area = max(0.0, float(w) * float(h))
        s += min(0.15, area / (1280 * 800) * 0.8)
        if prefer == "topmost":
            s += max(0.0, 0.12 - (float(y) / 800.0) * 0.12)
        elif prefer == "largest":
            s += min(0.12, area / (1280 * 800) * 1.0)
        if area < 200:
            s -= 0.05
    except Exception:
        pass

    return float(s)


# ============================================================
# Orchestrator
# ============================================================

class AgentOrchestrator:
    def __init__(self, model: str, cheap_model: str, user_data_dir: str, slow_mo_ms: int):
        self.browser = BrowserController(user_data_dir=user_data_dir, slow_mo_ms=slow_mo_ms)
        self.llm_main = OpenAIReActClient(model=model)
        self.llm_cheap = OpenAIReActClient(model=cheap_model)
        self.memory = Memory()

        self._last_observation: Dict[str, Any] = {}
        self._current_task: str = ""
        self._screens_dir = os.path.join(user_data_dir, "screens")
        self._vision_model = os.getenv("OPENAI_VISION_MODEL") or None
        self._auto_reload = os.getenv("AUTO_RELOAD_ON_STUCK", "1") == "1"
        self._last_reload_ts = 0.0
        self._handoff_enabled = os.getenv("ENABLE_HANDOFF", "1") == "1"
        self._awaiting_confirmation = False
        self._pending_task: Optional[str] = None
        self._last_user_task: Optional[str] = None
        self._task_id = 0
        self._observe_needed = True
        self._blind_steps = 0
        self._max_blind_steps = int(os.getenv("OBSERVE_MAX_BLIND_STEPS", "3"))
        self._auto_vision_on_stuck = os.getenv("AUTO_VISION_ON_STUCK", "1") == "1"
        self._last_vision_fp: Optional[str] = None

        # readiness + loop guard wrapper for observe
        self.stable_observer = StableObserver()

    def _derive_vision_query(self, task: str, obs: Dict[str, Any]) -> Optional[str]:
        t = (task or "").strip()
        if not t:
            return None
        # 1) Quoted phrase has priority
        m = re.search(r"[\"«](.+?)[\"»]", t)
        if m:
            q = m.group(1).strip()
            if q:
                return q
        # 2) Look for common action verbs and take following words
        verbs = ("найди", "поиск", "search", "открой", "нажми", "кликни", "перейди", "add", "delete", "удали")
        low = t.lower()
        for v in verbs:
            idx = low.find(v)
            if idx != -1:
                tail = t[idx + len(v):].strip()
                # take up to 4 words
                words = [w for w in re.split(r"[\s,;:]+", tail) if w]
                if words:
                    return " ".join(words[:4]).strip()
        # 3) Handoff queries as fallback
        handoff = obs.get("handoff") or {}
        queries = handoff.get("queries") or []
        if queries:
            return str(queries[0]).strip()
        return None

    def _is_simple_task(self, task: str) -> bool:
        t = (task or "").lower().strip()
        if not t:
            return False
        # Avoid risky or multi-step tasks
        if any(k in t for k in ("удали", "оплат", "куп", "закаж", "перевед", "подтверд", "вход", "логин")):
            return False
        if any(k in t for k in (" и ", " затем ", " потом ", " после ", ";", ",")):
            return False
        # Very short, single intent commands (open/go/back/reload)
        verbs = ("открой", "открыть", "перейди", "зайди", "назад", "вернись", "обнови", "перезагрузи")
        if not any(t.startswith(v) or f" {v} " in t for v in verbs):
            return False
        # Heuristic: short tasks only
        if len(t.split()) > 6:
            return False
        return True

    def _should_use_web_search(self, task: str) -> bool:
        if os.getenv("ENABLE_WEB_SEARCH", "1") != "1":
            return False
        t = (task or "").lower()
        if not t:
            return False
        # Don't use for trivial single-step tasks
        if self._is_simple_task(task):
            return False
        # Multi-step indicators
        multi = any(k in t for k in (" и ", " затем ", " потом ", " после ", ";"))
        # Site/UI indicators
        site = any(k in t for k in ("сайт", "страниц", "веб", "в браузере", "в браузер", "на сайте", "url", "http"))
        # Action verbs that often benefit from UI hints
        actions = any(k in t for k in ("найди", "поиск", "добавь", "удали", "оформи", "оплати", "закаж", "корзин"))
        return multi or (site and actions)

    async def start(self):
        await self.browser.start()
        os.makedirs(self._screens_dir, exist_ok=True)

    async def stop(self):
        await self.browser.stop()

    async def run_task(self, user_task: str):
        # New task: reset memory and loop guard unless user explicitly asked to continue
        if not self._is_user_continue(user_task) and not self._awaiting_confirmation:
            self._task_id += 1
            self.memory = Memory()
            self._last_observation = {}
            self.stable_observer = StableObserver()
            self._observe_needed = True
            self._blind_steps = 0

        # If user just confirms, continue pending task
        if self._awaiting_confirmation and self._is_user_confirmation(user_task):
            if self._pending_task:
                user_task = (
                    f"{self._pending_task}\n"
                    f"Пользователь подтвердил: {user_task}\n"
                    "Продолжай выполнение в том же контексте."
                )
            self._awaiting_confirmation = False
        # If user asks to continue, restate the last task
        elif self._is_user_continue(user_task) and self._last_user_task:
            user_task = (
                f"Продолжай предыдущую задачу: {self._last_user_task}\n"
                f"Пользователь попросил продолжить: {user_task}\n"
                "Не задавай повторных уточнений, действуй."
            )

        self._current_task = user_task
        input_items: List[Dict[str, Any]] = [
            {"role": "user", "content": f"Задача: {user_task}"},
            {"role": "user", "content": "Начни с observe(), затем действуй автономно."},
        ]

        loop = asyncio.get_running_loop()

        def tool_exec(name: str, args: dict):
            # Run async tool in main loop (thread-safe) from the worker thread.
            fut = asyncio.run_coroutine_threadsafe(self._tool_exec_async(name, args), loop)
            result = fut.result()

            # Console tool log (like "Using tool ...")
            console.print(f"[dim]Using tool:[/dim] [bold]{name}[/bold]")
            console.print(f"[dim]Input:[/dim] {args}")
            if name == "observe" and isinstance(result, dict):
                compact = {
                    "url": result.get("url"),
                    "title": result.get("title"),
                    "summary": (result.get("visible_text") or "")[:240],
                    "readiness": result.get("readiness"),
                    "loop_guard": result.get("loop_guard"),
                }
                if result.get("handoff"):
                    compact["handoff"] = result.get("handoff")
                if result.get("vision_hint"):
                    compact["vision_hint"] = result.get("vision_hint")
                console.print(f"[dim]Result:[/dim] {compact}\n")
            else:
                console.print(f"[dim]Result:[/dim] {result}\n")

            return result

        llm = self.llm_cheap if self._is_simple_task(user_task) else self.llm_main
        tools = TOOLS.copy()
        if self._should_use_web_search(user_task):
            tools.append(WEB_SEARCH_TOOL)
        text = await asyncio.to_thread(
            llm.run_tool_loop,
            system_instructions=pick_instructions(),
            tools=tools,
            input_items=input_items,
            tool_executor=tool_exec,
            max_iters=35,
            parallel_tool_calls=False,
        )

        console.print("\n[bold yellow]Результат:[/bold yellow]")
        console.print(text)

        # Detect if assistant asked for confirmation in text (fallback)
        if self._looks_like_confirmation_request(text):
            self._awaiting_confirmation = True
            self._pending_task = self._current_task
        else:
            self._awaiting_confirmation = False
            self._pending_task = None
            # Save last non-trivial task for "continue" prompts
            if not self._is_user_continue(self._current_task):
                self._last_user_task = self._current_task

    def _looks_like_confirmation_request(self, text: str) -> bool:
        t = (text or "").lower()
        if "подтверд" not in t:
            return False
        triggers = ("пожалуйста", "напишите", "подтверждение", "если вы подтверждаете", "подтвердите")
        return any(x in t for x in triggers)

    def _is_user_confirmation(self, text: str) -> bool:
        t = (text or "").strip().lower()
        # normalize punctuation
        t = t.translate(str.maketrans("", "", string.punctuation + "“”«»"))
        if any(k in t for k in ("подтверждаю", "подтверждаю", "подтвердить", "подтверждение", "согласен")):
            return True
        # simple "да" as confirmation when we're waiting for it
        return t in ("да", "yes", "y")

    def _is_user_continue(self, text: str) -> bool:
        t = (text or "").strip().lower()
        t = t.translate(str.maketrans("", "", string.punctuation + "“”«»"))
        return t in (
            "приступай",
            "продолжай",
            "продолжи",
            "давай",
            "поехали",
            "далее",
            "continue",
            "go on",
            "start",
        )

    async def _tool_exec_async(self, name: str, args: dict):
        # Security layer
        if is_risky(name, args):
            ok = ask_user_confirmation(name, args)
            if not ok:
                return {"ok": False, "error": "User denied risky action."}

        try:
            if name == "observe":
                # If we recently observed and no trigger demands a new one, return cached minimal data
                if not self._observe_needed and self._last_observation and self._blind_steps < self._max_blind_steps:
                    return {
                        "url": self._last_observation.get("url"),
                        "title": self._last_observation.get("title"),
                        "summary": "observe skipped (cooldown) — no strong trigger since last observe.",
                        "cached": True,
                    }
                # ✅ stable observe: waits for SPA readiness and detects loops
                obs = await self.stable_observer.observe_stable(self._observe, self.browser.wait)
                # Log readiness details
                readiness = obs.get("readiness") or {}
                loop_guard = obs.get("loop_guard") or {}
                console.print(
                    f"[dim]Readiness:[/dim] {readiness} | [dim]Loop:[/dim] {loop_guard}\n"
                )
                # Auto-reload when page is stuck or shows JS-disabled fallback
                if self._auto_reload:
                    reason = (obs.get("readiness") or {}).get("reason", "")
                    stuck = (obs.get("loop_guard") or {}).get("stuck", False)
                    now = time.time()
                    should_reload = (
                        ("javascript" in str(reason).lower())
                        or (reason == "max_attempts_exceeded" and stuck)
                    )
                    if should_reload and (now - self._last_reload_ts) > 15:
                        self._last_reload_ts = now
                        await self.browser.reload()
                        obs = await self.stable_observer.observe_stable(self._observe, self.browser.wait)

                # Auto vision hint after repeated stuck snapshots
                loop_guard = obs.get("loop_guard") or {}
                if (
                    self._auto_vision_on_stuck
                    and loop_guard.get("stuck")
                    and loop_guard.get("streak", 0) >= 2
                ):
                    fp = snapshot_fingerprint(obs)
                    if fp != self._last_vision_fp:
                        q = self._derive_vision_query(self._current_task, obs)
                        if q:
                            try:
                                vision = await self._vision_find({"query": q})
                                obs["vision_hint"] = {"query": q, **vision}
                            except Exception:
                                obs["vision_hint"] = {"query": q, "ok": False, "error": "vision_failed"}
                        self._last_vision_fp = fp

                # Handoff: helper analysis for complex pages
                if self._handoff_enabled:
                    helper = self.llm_main.helper_analyze_observation(task=self._current_task, observation=obs)
                    if helper.get("ok"):
                        obs["handoff"] = helper.get("data")
                self._last_observation = obs
                self._observe_needed = False
                self._blind_steps = 0
                return obs

            if name == "extract_text":
                max_chars = int(args.get("max_chars", 4000))
                text = await get_visible_text(self.browser.page, max_chars=max_chars)
                return {"ok": True, "text": text, "len": len(text)}

            if name == "extract_links":
                limit = int(args.get("limit", 60))
                links = await collect_links(self.browser.page, limit=limit)
                return {"ok": True, "count": len(links), "links": links}

            if name == "extract_list":
                return await self._extract_list(args)

            if name == "goto":
                await self.browser.goto(args["url"])
                self.memory.add_step({"tool": "goto", "args": args, "result": "ok"})
                self._observe_needed = True
                self._blind_steps = 0
                extra = await self._after_action("goto")
                return {"ok": True, **extra}

            if name == "click":
                idx = int(args.get("index", 0))
                try:
                    pre = (await self.browser.get_url_title()).get("url")
                    await self.browser.click(args["element_id"], index=idx)
                    await self._maybe_follow_href(args["element_id"], pre)
                    self.memory.add_step({"tool": "click", "args": args, "result": "ok"})
                    self._blind_steps += 1
                    extra = await self._after_action("click")
                    return {"ok": True, **extra}
                except Exception as e:
                    # Fallback: click by coordinates if DOM click failed or element is blocked
                    ok = await self._click_at_element_center(args["element_id"])
                    if ok:
                        await self._maybe_follow_href(args["element_id"], pre)
                        self.memory.add_step({"tool": "click", "args": args, "result": "ok(click_at_fallback)"})
                        self._blind_steps += 1
                        extra = await self._after_action("click_at")
                        return {"ok": True, "via": "click_at", **extra}
                    raise e

            if name == "type":
                idx = int(args.get("index", 0))
                clear = bool(args.get("clear", True))
                await self.browser.type(args["element_id"], args["text"], clear=clear, index=idx)
                # mask text in memory
                mem_args = {**args, "text": "***"}
                self.memory.add_step({"tool": "type", "args": mem_args, "result": "ok"})
                if bool(args.get("submit", False)):
                    await self.browser.press("Enter")
                    self.memory.add_step({"tool": "press", "args": {"key": "Enter"}, "result": "ok"})
                    self._observe_needed = True
                    self._blind_steps = 0
                else:
                    self._blind_steps += 1
                extra = await self._after_action("type")
                return {"ok": True, **extra}

            if name == "press":
                await self.browser.press(args["key"])
                self.memory.add_step({"tool": "press", "args": args, "result": "ok"})
                if str(args.get("key", "")).lower() in ("enter", "return"):
                    self._observe_needed = True
                    self._blind_steps = 0
                else:
                    self._blind_steps += 1
                extra = await self._after_action("press")
                return {"ok": True, **extra}

            if name == "scroll":
                await self.browser.scroll(dy=int(args.get("dy", 700)))
                self.memory.add_step({"tool": "scroll", "args": args, "result": "ok"})
                self._blind_steps += 1
                extra = await self._after_action("scroll")
                return {"ok": True, **extra}

            if name == "back":
                await self.browser.back()
                self.memory.add_step({"tool": "back", "args": args, "result": "ok"})
                self._observe_needed = True
                self._blind_steps = 0
                extra = await self._after_action("back")
                return {"ok": True, **extra}

            if name == "wait":
                await self.browser.wait(ms=int(args.get("ms", 800)))
                self.memory.add_step({"tool": "wait", "args": args, "result": "ok"})
                extra = await self._after_action("wait")
                return {"ok": True, **extra}

            if name == "reload":
                await self.browser.reload()
                self.memory.add_step({"tool": "reload", "args": args, "result": "ok"})
                self._observe_needed = True
                self._blind_steps = 0
                extra = await self._after_action("reload")
                return {"ok": True, **extra}

            if name == "click_best_match":
                out = await self._click_best_match(args)
                if out.get("ok"):
                    extra = await self._after_action("click_best_match")
                    return {**out, **extra}
                return out

            if name == "click_at":
                await self.browser.click_at(int(args["x"]), int(args["y"]))
                self.memory.add_step({"tool": "click_at", "args": args, "result": "ok"})
                extra = await self._after_action("click_at")
                return {"ok": True, **extra}

            if name == "vision_find":
                return await self._vision_find(args)

            if name == "find_elements":
                return await self._find_elements(args)

            return {"ok": False, "error": f"Unknown tool: {name}"}

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            self.memory.add_step({"tool": name, "args": args, "result": msg})
            return {
                "ok": False,
                "error": msg,
                "recovery_hint": (
                    "Попробуй: observe(); если элемент вне экрана — используй поиск/меню страницы; "
                    "если попап — press Escape; затем снова observe() и выбери другой элемент."
                ),
            }

    async def _click_best_match(self, args: dict) -> Dict[str, Any]:
        # Ensure fresh observation
        if not self._last_observation.get("elements"):
            self._last_observation = await self.stable_observer.observe_stable(self._observe, self.browser.wait)

        query = args["query"]
        queries = args.get("queries") or []
        vision_fallback = bool(args.get("vision_fallback", True))
        prefer = args.get("prefer", "default")
        role_hint = args.get("role_hint")

        els = self._last_observation.get("elements", [])
        if not els:
            if vision_fallback:
                vf = await self._vision_find({"query": query})
                if vf.get("ok") and vf.get("confidence", 0.0) >= 0.35:
                    await self.browser.click_at(int(vf["x"]), int(vf["y"]))
                    self.memory.add_step(
                        {"tool": "click_best_match", "args": {"query": query}, "result": "ok(vision_click)"}
                    )
                    return {"ok": True, "via": "vision_click", **vf}
            return {"ok": False, "error": "No elements in observation. Try observe()."}

        all_queries = [query] + [q for q in queries if q and q != query]
        best_score = -1.0
        best = None
        best_query = query
        best_scored: List[Tuple[float, dict]] = []
        for q in all_queries:
            scored: List[Tuple[float, dict]] = [(_score_element(el, q, role_hint, prefer), el) for el in els]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored and scored[0][0] > best_score:
                best_score, best = scored[0]
                best_query = q
                best_scored = scored

        if best is None or best_score < 0.25:
            if vision_fallback:
                vf = await self._vision_find({"query": best_query})
                if vf.get("ok") and vf.get("confidence", 0.0) >= 0.35:
                    await self.browser.click_at(int(vf["x"]), int(vf["y"]))
                    self.memory.add_step(
                        {"tool": "click_best_match", "args": {"query": best_query}, "result": "ok(vision_click)"}
                    )
                    return {"ok": True, "via": "vision_click", **vf}
            return {
                "ok": False,
                "error": "Low confidence best match",
                "best_candidates": [
                    {
                        "score": float(s),
                        "id": e.get("id"),
                        "tag": e.get("tag"),
                        "role": e.get("role"),
                        "name": e.get("name"),
                        "text": e.get("text"),
                        "href": e.get("href"),
                        "disabled": bool(e.get("disabled", False)),
                        "bbox": e.get("bbox"),
                    }
                    for s, e in best_scored[:8]
                ],
                "hint": (
                    "Уточни query (добавь часть названия/канал/длительность/уточняющий текст) "
                    "или сделай scroll() и observe() чтобы увидеть нужный блок."
                ),
            }

        pre = (await self.browser.get_url_title()).get("url")
        try:
            await self.browser.click(best["id"], index=0)
        except Exception:
            ok = await self._click_at_element_center(best["id"])
            if not ok:
                raise
        await self._maybe_follow_href(best["id"], pre)
        self.memory.add_step(
            {
                "tool": "click_best_match",
                "args": {"query": best_query, "prefer": prefer, "role_hint": role_hint},
                "result": f"clicked {best.get('id')} score={best_score:.2f}",
            }
        )

        return {
            "ok": True,
            "clicked": {
                "id": best.get("id"),
                "tag": best.get("tag"),
                "role": best.get("role"),
                "name": best.get("name"),
                "text": best.get("text"),
                "href": best.get("href"),
                "bbox": best.get("bbox"),
            },
            "score": float(best_score),
            "top5": [
                {"score": float(s), "id": e.get("id"), "name": e.get("name"), "text": e.get("text"), "href": e.get("href")}
                for s, e in scored[:5]
            ],
        }

    async def _after_action(self, tool_name: str) -> Dict[str, Any]:
        """
        Сделать скриншот после действия.
        """
        try:
            # Give UI a moment to settle before screenshot
            await self.browser.wait(ms=1500)
            os.makedirs(self._screens_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            path = os.path.join(self._screens_dir, f"{tool_name}_{ts}.png")
            await self.browser.screenshot(path)
            return {"screenshot": path}
        except Exception as e:
            return {"screenshot_error": f"{type(e).__name__}: {e}"}

    async def _maybe_follow_href(self, element_id: str, pre_url: Optional[str]) -> None:
        """
        If click didn't change URL but element has href, follow it as a fallback.
        """
        try:
            post = (await self.browser.get_url_title()).get("url")
            if not pre_url or not post or pre_url != post:
                return
            # Prefer href from element snapshot if present in last observation
            for e in (self._last_observation.get("elements") or []):
                if e.get("id") == element_id and e.get("href"):
                    await self.browser.goto(e.get("href"))
                    self.memory.add_step(
                        {"tool": "goto", "args": {"url": e.get("href")}, "result": "ok(follow_href)"}
                    )
                    return
        except Exception:
            return

    async def _click_at_element_center(self, element_id: str) -> bool:
        el = self.browser.element_index.get(element_id)
        if not el or not el.bbox:
            return False
        try:
            x, y, w, h = el.bbox
            cx = int(x + (w / 2))
            cy = int(y + (h / 2))
        except Exception:
            return False
        try:
            await self.browser.click_at(cx, cy)
            return True
        except Exception:
            return False

    async def _vision_find(self, args: dict) -> Dict[str, Any]:
        query = (args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "query is required"}

        os.makedirs(self._screens_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        path = os.path.join(self._screens_dir, f"vision_find_{ts}.png")
        await self.browser.screenshot(path)

        try:
            vp = {"width": 1280, "height": 800}
            out = self.llm_main.vision_locate(query=query, image_path=path, viewport=vp, model=self._vision_model)
            if not out.get("ok"):
                return {"ok": False, "error": out.get("error"), "raw": out.get("raw"), "screenshot": path}

            data = out.get("data") or {}
            return {
                "ok": True,
                "screenshot": path,
                "x": int(data.get("x", 0)),
                "y": int(data.get("y", 0)),
                "confidence": float(data.get("confidence", 0.0)),
                "reason": data.get("reason"),
            }
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "screenshot": path}

    async def _find_elements(self, args: dict) -> Dict[str, Any]:
        query = (args.get("query") or "").strip()
        role = (args.get("role") or "").strip().lower()
        tag = (args.get("tag") or "").strip().lower()
        limit = int(args.get("limit", 20))

        raw = await collect_raw_elements(self.browser.page, limit=800)
        elements = normalize_elements(raw)

        if role:
            elements = [e for e in elements if (e.get("role") or "").lower() == role]
        if tag:
            elements = [e for e in elements if (e.get("tag") or "").lower() == tag]

        if not elements:
            return {"ok": False, "error": "No elements found", "count": 0}

        if query:
            scored = [(_score_element(e, query, role_hint=None, prefer="default"), e) for e in elements]
        else:
            def base_score(el: dict) -> float:
                bbox = el.get("bbox") or [0, 0, 0, 0]
                try:
                    _, y, w, h = bbox
                    area = max(0.0, float(w) * float(h))
                except Exception:
                    area = 0.0
                    y = 0.0

                s = min(0.3, area / (1280 * 800))
                s += max(0.0, 0.15 - (float(y) / 800.0) * 0.15)
                if el.get("role") in ("row", "listitem", "option", "menuitem", "tab", "checkbox", "gridcell"):
                    s += 0.15
                if el.get("tag") in ("input", "textarea", "select"):
                    s += 0.2
                if el.get("tag") in ("button", "a") or el.get("role") in ("button", "link"):
                    s += 0.1
                return float(s)

            scored = [(base_score(e), e) for e in elements]

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = scored[: max(1, limit)]

        for _, e in picked:
            self.browser.element_index[e["id"]] = Element(
                id=e["id"],
                role=e.get("role"),
                name=e.get("name"),
                tag=e.get("tag"),
                text=e.get("text"),
                bbox=e.get("bbox"),
                locator=e.get("locator"),
            )

        return {
            "ok": True,
            "count": len(picked),
            "elements": [
                {
                    "score": float(s),
                    "id": e.get("id"),
                    "tag": e.get("tag"),
                    "role": e.get("role"),
                    "name": e.get("name"),
                    "text": e.get("text"),
                    "href": e.get("href"),
                    "title": e.get("title"),
                    "tooltip": e.get("tooltip"),
                    "disabled": bool(e.get("disabled", False)),
                    "bbox": e.get("bbox"),
                }
                for s, e in picked
            ],
        }

    async def _extract_list(self, args: dict) -> Dict[str, Any]:
        role = (args.get("role") or "").strip().lower()
        tag = (args.get("tag") or "").strip().lower()
        limit = int(args.get("limit", 30))

        raw = await collect_raw_elements(self.browser.page, limit=800)
        elements = normalize_elements(raw)

        if role:
            elements = [e for e in elements if (e.get("role") or "").lower() == role]
        if tag:
            elements = [e for e in elements if (e.get("tag") or "").lower() == tag]

        # Keep only elements with some text/name
        elements = [e for e in elements if (e.get("name") or e.get("text"))]
        elements = elements[: max(1, limit)]

        for e in elements:
            self.browser.element_index[e["id"]] = Element(
                id=e["id"],
                role=e.get("role"),
                name=e.get("name"),
                tag=e.get("tag"),
                text=e.get("text"),
                bbox=e.get("bbox"),
                locator=e.get("locator"),
            )

        return {
            "ok": True,
            "count": len(elements),
            "items": [
                {
                    "id": e.get("id"),
                    "tag": e.get("tag"),
                    "role": e.get("role"),
                    "name": e.get("name"),
                    "text": e.get("text"),
                    "href": e.get("href"),
                    "title": e.get("title"),
                    "tooltip": e.get("tooltip"),
                    "bbox": e.get("bbox"),
                }
                for e in elements
            ],
        }

    async def _observe(self) -> Dict[str, Any]:
        """
        Raw observe: builds element index (for click/type) and returns compact observation for LLM.
        StableObserver wraps this to wait for SPA readiness + detect loops.
        """
        meta = await self.browser.get_url_title()
        visible_text, raw_elements = await perceive_page(self.browser.page)
        elements = normalize_elements(raw_elements)

        # Fill index for tool actions
        self.browser.element_index.clear()
        for e in elements:
            self.browser.element_index[e["id"]] = Element(
                id=e["id"],
                role=e.get("role"),
                name=e.get("name"),
                tag=e.get("tag"),
                text=e.get("text"),
                bbox=e.get("bbox"),
                locator=e.get("locator"),
            )

        # Rank elements to keep context compact but useful
        def rank(el: dict) -> float:
            if el.get("disabled"):
                return -999.0
            tag = (el.get("tag") or "").lower()
            role = (el.get("role") or "").lower()
            name = (el.get("name") or "").lower()
            text = (el.get("text") or "").lower()
            href = (el.get("href") or "").lower()

            s = 0.0
            if tag in ("input", "textarea", "select"):
                s += 3.0
            if tag in ("button", "a") or role in ("button", "link"):
                s += 1.0
            if role in ("row", "listitem", "option", "menuitem", "tab", "checkbox", "gridcell"):
                s += 1.2
            if href:
                s += 0.2
            if "search" in (name + " " + text) or "поиск" in (name + " " + text):
                s += 2.0

            bbox = el.get("bbox") or [0, 0, 0, 0]
            try:
                _, y, w, h = bbox
                area = max(0.0, float(w) * float(h))
                s += min(0.15, area / (1280 * 800))
                s += max(0.0, 0.08 - (float(y) / 800.0) * 0.08)
            except Exception:
                pass
            return s

        elements_sorted = sorted(elements, key=rank, reverse=True)[:60]

        def clip(v: Any, n: int) -> Optional[str]:
            if v is None:
                return None
            s = str(v).strip()
            return s[:n]

        obs_elements: List[Dict[str, Any]] = []
        for e in elements_sorted:
            obs_elements.append(
                {
                    "id": e["id"],
                    "tag": e.get("tag"),
                    "role": e.get("role"),
                    "name": clip(e.get("name"), 80),
                    "text": clip(e.get("text"), 90),
                    "href": clip(e.get("href"), 140),
                    "title": clip(e.get("title"), 80),
                    "tooltip": clip(e.get("tooltip"), 90),
                    "disabled": bool(e.get("disabled", False)),
                    "bbox": e.get("bbox"),
                }
            )

        return {
            "url": meta["url"],
            "title": meta["title"],
            "visible_text": (visible_text or "")[:1000],
            "elements": obs_elements,
            "memory": self.memory.dump_for_prompt(),
            "notes": (
                "Если элементов мало/страница не готова — observe() автоматически подождёт. "
                "Для дизамбигуации используй click_best_match()."
            ),
            "startup_hint": (
                "Пустая страница. Не ищи элементы — выполни goto на нужный сайт. "
                "Если адрес неизвестен, используй web_search и затем goto."
            )
            if (meta.get("url") in ("about:blank", "", None)) or (not obs_elements and not (visible_text or "").strip())
            else None,
        }

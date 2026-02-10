# src/agent/llm.py
import base64
import json
import time
import os
from typing import Any, Dict, List, Optional
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError

class OpenAIReActClient:
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model
        self.helper_model = os.getenv("OPENAI_HELPER_MODEL", "gpt-4.1-mini")

    def _create_with_retry(self, **kwargs):
        # простой backoff; обычно хватает
        max_attempts = 8
        delay = 0.8
        for attempt in range(1, max_attempts + 1):
            try:
                return self.client.responses.create(**kwargs)
            except RateLimitError as e:
                # 429: ждём и повторяем
                # В сообщении часто есть "Please try again in Xs"
                time.sleep(delay)
                delay = min(delay * 1.8, 8.0)
                if attempt == max_attempts:
                    raise
            except (APITimeoutError, APIError) as e:
                # временные ошибки — тоже можно ретраить
                time.sleep(delay)
                delay = min(delay * 1.8, 8.0)
                if attempt == max_attempts:
                    raise

    def run_tool_loop(
        self,
        *,
        system_instructions: str,
        tools: List[Dict[str, Any]],
        input_items: List[Dict[str, Any]],
        tool_executor,
        max_iters: int = 25,
        parallel_tool_calls: bool = False,
    ) -> str:
        def _item_type(it) -> Optional[str]:
            return getattr(it, "type", it.get("type") if isinstance(it, dict) else None)

        def _item_call_id(it) -> Optional[str]:
            return getattr(it, "call_id", it.get("call_id") if isinstance(it, dict) else None)

        def _prune_input_items(items: List[Dict[str, Any]], keep_last: int = 24) -> None:
            if len(items) <= 2 + keep_last:
                return
            head = items[:2]
            tail = items[-keep_last:]

            # Ensure every function_call_output has its matching function_call in the kept window.
            needed_call_ids = {
                _item_call_id(it)
                for it in tail
                if _item_type(it) == "function_call_output" and _item_call_id(it)
            }
            extras: List[Dict[str, Any]] = []
            if needed_call_ids:
                for i, it in enumerate(items):
                    if _item_type(it) == "function_call" and _item_call_id(it) in needed_call_ids:
                        # include preceding reasoning if present
                        if i > 0 and _item_type(items[i - 1]) == "reasoning":
                            extras.append(items[i - 1])
                        extras.append(it)

            # Ensure reasoning preceding kept messages/web_search is preserved (required by some models)
            tail_ids = {id(it) for it in tail}
            for i, it in enumerate(items):
                if _item_type(it) == "reasoning":
                    if i + 1 < len(items) and id(items[i + 1]) in tail_ids and _item_type(items[i + 1]) in ("message", "web_search"):
                        extras.append(it)

            # Deduplicate while preserving order
            seen_ids = set()
            merged: List[Dict[str, Any]] = []
            for part in (head, extras, tail):
                for it in part:
                    key = (id(it), _item_type(it), _item_call_id(it))
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    merged.append(it)

            items[:] = merged

        def _should_force_action(text: str) -> bool:
            t = (text or "").lower()
            # If assistant is asking for user action, don't force
            if any(k in t for k in ("подтверд", "введите", "нужен", "нужна", "2fa", "капч", "captcha", "логин")):
                return False
            # If it's just narrating / stalling, force to act
            return any(
                k in t
                for k in (
                    "готов приступить",
                    "продолжу работу",
                    "продолжу выполнение",
                    "нужно уточнить",
                    "нужна цель",
                    "могу продолжить",
                    "приступлю после",
                )
            )

        last_tool_name: Optional[str] = None
        for _ in range(max_iters):
            resp = self._create_with_retry(
                model=self.model,
                instructions=system_instructions,
                tools=tools,
                input=input_items,
                parallel_tool_calls=parallel_tool_calls,

                max_output_tokens=420,
            )

            safe_out = []
            i = 0
            while i < len(resp.output):
                item = resp.output[i]
                if item.type == "message":
                    safe_out.append(item)
                    i += 1
                    continue
                if item.type == "reasoning":
                    # сохраняем reasoning только если сразу за ним идет function_call или message
                    if i + 1 < len(resp.output) and resp.output[i + 1].type in ("function_call", "message", "web_search"):
                        safe_out.append(item)
                        safe_out.append(resp.output[i + 1])
                        i += 2
                        continue
                    # иначе пропускаем reasoning
                    i += 1
                    continue
                if item.type == "function_call":
                    safe_out.append(item)
                    i += 1
                    continue
                if item.type == "web_search":
                    safe_out.append(item)
                    i += 1
                    continue
                i += 1

            input_items += safe_out

            tool_calls = [item for item in safe_out if item.type == "function_call"]
            if not tool_calls:
                text = resp.output_text or ""
                t = text.strip()
                if not t:
                    input_items.append(
                        {
                            "role": "user",
                            "content": "Ты не сделал действий. Продолжай и вызывай инструменты, пока задача не выполнена.",
                        }
                    )
                    continue
                if last_tool_name in ("observe", "find_elements", "vision_find") and (
                    "готово" in t.lower() or "следующ" in t.lower() or len(t) < 80
                ):
                    input_items.append(
                        {
                            "role": "user",
                            "content": "Не заканчивай. Ты только что наблюдал страницу. Выполни следующее действие через инструмент.",
                        }
                    )
                    continue
                if _should_force_action(text):
                    input_items.append(
                        {
                            "role": "user",
                            "content": "Не описывай и не уточняй. Действуй через инструменты и выполни задачу до конца.",
                        }
                    )
                    continue
                return text

            for call in tool_calls:
                name = call.name
                raw_args = call.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    # Fallback to empty args and report parse error to the model
                    args = {}
                    input_items.append(
                        {
                            "role": "user",
                            "content": f"Некорректные аргументы tool-call для {name}: {raw_args!r}. Повтори вызов инструмента с валидным JSON.",
                        }
                    )
                    continue
                out = tool_executor(name, args)
                last_tool_name = name

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(out, ensure_ascii=False),
                    }
                )
                _prune_input_items(input_items, keep_last=24)
        return "Не удалось завершить задачу: превышен лимит итераций."

    def vision_locate(
        self,
        *,
        query: str,
        image_path: str,
        viewport: Optional[Dict[str, int]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        vp = viewport or {"width": 1280, "height": 800}
        prompt = (
            "Ты видишь скриншот веб-страницы. Найди на нем элемент по запросу пользователя. "
            "Верни JSON строго в формате: "
            '{"x": int, "y": int, "confidence": 0.0-1.0, "reason": "short"}. '
            f"Координаты в пикселях относительно верхнего левого угла. "
            f"Размеры изображения: {vp.get('width')}x{vp.get('height')}. "
            f"Запрос: {query}"
        )

        resp = self._create_with_retry(
            model=model or self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            max_output_tokens=120,
        )

        text = (resp.output_text or "").strip()
        try:
            data = json.loads(text)
            return {"ok": True, "data": data}
        except Exception:
            return {"ok": False, "error": "Failed to parse JSON", "raw": text}

    def helper_analyze_observation(self, *, task: str, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lightweight helper: summarize page state and suggest next actions/queries.
        """
        try:
            url = observation.get("url")
            title = observation.get("title")
            visible_text = (observation.get("visible_text") or "")[:1200]
            elements = observation.get("elements") or []
            elements = elements[:20]

            payload = {
                "url": url,
                "title": title,
                "visible_text": visible_text,
                "elements": [
                    {
                        "id": e.get("id"),
                        "tag": e.get("tag"),
                        "role": e.get("role"),
                        "name": e.get("name"),
                        "text": e.get("text"),
                        "href": e.get("href"),
                    }
                    for e in elements
                ],
            }

            prompt = (
                "Ты помощник. По задаче пользователя и наблюдению страницы верни JSON:\n"
                "{\n"
                '  "summary": "1-2 коротких предложения о текущем состоянии",\n'
                '  "next_actions": ["кратко: что сделать дальше (1-3)"],\n'
                '  "queries": ["краткие query для click_best_match"],\n'
                '  "confidence": 0.0-1.0\n'
                "}\n"
                "Не добавляй лишний текст, только JSON."
            )

            resp = self._create_with_retry(
                model=self.helper_model,
                input=[
                    {"role": "user", "content": f"Задача: {task}"},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=200,
            )

            text = (resp.output_text or "").strip()
            data = json.loads(text)
            return {"ok": True, "data": data}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

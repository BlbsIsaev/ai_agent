import asyncio
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import async_playwright, Browser, Page


INTERACTIVE_ROLES = [
    "button",
    "link",
    "textbox",
    "searchbox",
    "combobox",
    "checkbox",
    "radio",
    "menuitem",
    "tab",
]


@dataclass
class Candidate:
    id: str
    role: str
    name: str
    disabled: Optional[bool] = None
    checked: Optional[bool] = None
    expanded: Optional[bool] = None
    nth: int = 0  # index among same (role+name)


@dataclass
class PageState:
    url: str
    title: str
    aria_snapshot: str
    candidates: List[Candidate]


class BrowserController:
    """
    Headful Playwright controller.
    ARIA-first observe() returns aria snapshot + compact list of interactive candidates.
    """

    def __init__(self, headless: bool = False):
        self.headless = headless
        self._pw = None
        self._browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        # last observe cache: id -> (role, name, nth)
        self._last_index: Dict[str, Tuple[str, str, int]] = {}

    async def start(self) -> None:
        print("[start] starting playwright...")
        self._pw = await async_playwright().start()

        print("[start] launching chromium...")
        self._browser = await self._pw.chromium.launch(
            headless=self.headless,
            timeout=15000,   # важно: чтобы не висело вечно
        )

        print("[start] creating context/page...")
        context = await self._browser.new_context()
        self.page = await context.new_page()

        print("[start] goto about:blank")
        await self.page.goto("about:blank")
        print("[start] done")


    async def close(self) -> None:
        try:
            if self._browser:
                await self._browser.close()
        finally:
            if self._pw:
                await self._pw.stop()

    async def goto(self, url: str) -> None:
        assert self.page is not None
        await self.page.goto(url, wait_until="domcontentloaded")

    async def observe(self, max_candidates: int = 40) -> PageState:
        """
        Returns:
          - aria_snapshot (Playwright aria_snapshot YAML string)
          - candidates: role/name/state condensed
        """
        assert self.page is not None

        # Best-effort: some sites may throw; handle gracefully.
        try:
            snapshot = await self.page.aria_snapshot()
        except Exception as e:
            snapshot = f"# aria_snapshot unavailable: {type(e).__name__}: {e}"

        url = self.page.url
        title = await self.page.title()

        candidates: List[Candidate] = []
        role_name_seen: Dict[Tuple[str, str], int] = {}

        self._last_index.clear()
        cnum = 0

        for role in INTERACTIVE_ROLES:
            if cnum >= max_candidates:
                break

            locator = self.page.get_by_role(role)
            try:
                count = await locator.count()
            except Exception:
                continue

            for i in range(min(count, max_candidates - cnum)):
                if cnum >= max_candidates:
                    break

                el = locator.nth(i)

                # Skip invisible elements (best-effort)
                try:
                    visible = await el.is_visible()
                    if not visible:
                        continue
                except Exception:
                    # if visibility check fails, keep going
                    pass

                # Try to get accessible name via aria snapshot signal:
                # Playwright doesn't directly expose accessible name from locator reliably across roles,
                # so we use a best-effort heuristic from DOM properties.
                name = await self._best_effort_name(el)

                name = (name or "").strip()
                if not name:
                    # If no name, still keep it, but it may be less useful
                    name = ""

                disabled = await self._safe_bool(el, "el => !!el.disabled")
                checked = await self._safe_bool(el, "el => (el.getAttribute('aria-checked') === 'true') || (el.checked === true)")
                expanded = await self._safe_bool(el, "el => (el.getAttribute('aria-expanded') === 'true')")

                key = (role, name)
                nth_same = role_name_seen.get(key, 0)
                role_name_seen[key] = nth_same + 1

                cid = f"c{cnum}"
                cnum += 1

                cand = Candidate(
                    id=cid,
                    role=role,
                    name=name,
                    disabled=disabled,
                    checked=checked,
                    expanded=expanded,
                    nth=nth_same,
                )
                candidates.append(cand)
                self._last_index[cid] = (role, name, nth_same)

        return PageState(url=url, title=title, aria_snapshot=snapshot, candidates=candidates)

    async def _best_effort_name(self, el) -> str:
        # Prefer aria-label, aria-labelledby text, label text, then innerText/value/placeholder.
        js = """
        (el) => {
          const byId = (id) => document.getElementById(id);
          const textOf = (node) => (node && node.innerText) ? node.innerText.trim() : "";
          const ariaLabel = el.getAttribute('aria-label') || "";
          if (ariaLabel.trim()) return ariaLabel;

          const labelledBy = el.getAttribute('aria-labelledby') || "";
          if (labelledBy.trim()) {
            const ids = labelledBy.split(/\\s+/).filter(Boolean);
            const parts = ids.map(id => textOf(byId(id))).filter(Boolean);
            if (parts.length) return parts.join(" ").trim();
          }

          // If input/select, try associated <label for="">
          if (el.id) {
            const lbl = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
            if (lbl && lbl.innerText.trim()) return lbl.innerText.trim();
          }
          // Or nearest wrapping label
          const wrapLabel = el.closest('label');
          if (wrapLabel && wrapLabel.innerText.trim()) return wrapLabel.innerText.trim();

          const placeholder = (el.getAttribute('placeholder') || "").trim();
          if (placeholder) return placeholder;

          const value = (el.value !== undefined ? String(el.value) : "").trim();
          if (value) return value;

          const txt = (el.innerText || "").trim();
          if (txt) return txt;

          const title = (el.getAttribute('title') || "").trim();
          if (title) return title;

          return "";
        }
        """
        try:
            return await el.evaluate(js)
        except Exception:
            return ""

    async def _safe_bool(self, el, js_expr: str) -> Optional[bool]:
        try:
            return bool(await el.evaluate(js_expr))
        except Exception:
            return None

    def get_last_target(self, cid: str) -> Optional[Tuple[str, str, int]]:
        return self._last_index.get(cid)

    def state_to_dict(self, state: PageState) -> Dict[str, Any]:
        return {
            "url": state.url,
            "title": state.title,
            "aria_snapshot": state.aria_snapshot,
            "candidates": [asdict(c) for c in state.candidates],
        }

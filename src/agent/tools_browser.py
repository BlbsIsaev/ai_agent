# src/agent/tools_browser.py
import os
import platform
from dataclasses import dataclass
from typing import Any, Dict, Optional

from playwright.async_api import async_playwright, BrowserContext, Page


@dataclass
class Element:
    id: str
    role: Optional[str]
    name: Optional[str]
    tag: Optional[str]
    text: Optional[str]
    bbox: Optional[list]
    locator: Dict[str, Any]  # recipe for building a locator


def detect_chrome_path() -> str | None:
    system = platform.system()

    candidates = []
    if system == "Darwin":  # macOS
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            os.path.expanduser("~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        ]
    elif system == "Linux":
        candidates = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
        ]
    elif system == "Windows":
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class BrowserController:
    def __init__(self, user_data_dir: str, slow_mo_ms: int = 120):
        self.user_data_dir = user_data_dir
        self.slow_mo_ms = slow_mo_ms
        self._pw = None
        self.ctx: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.element_index: Dict[str, Element] = {}

    async def start(self):
        os.makedirs(self.user_data_dir, exist_ok=True)
        self._pw = await async_playwright().start()

        chrome_path = os.getenv("CHROME_EXECUTABLE_PATH")
        chrome_path = os.path.expanduser(chrome_path) if chrome_path else None

        viewport = None
        if os.getenv("BROWSER_VIEWPORT_WIDTH") and os.getenv("BROWSER_VIEWPORT_HEIGHT"):
            viewport = {
                "width": int(os.getenv("BROWSER_VIEWPORT_WIDTH", "1280")),
                "height": int(os.getenv("BROWSER_VIEWPORT_HEIGHT", "800")),
            }

        launch_kwargs: Dict[str, Any] = dict(
            user_data_dir=self.user_data_dir,
            headless=False,
            slow_mo=self.slow_mo_ms,
            viewport=viewport,
            locale="ru-RU",
            timezone_id="Europe/Moscow",
            timeout=60_000,
            device_scale_factor=1,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-infobars",
                "--start-maximized",
            ],
            # try to remove the most obvious automation flag
            ignore_default_args=[
                "--enable-automation",
            ],
        )

        if chrome_path and os.path.exists(chrome_path):
            print(f"[browser] Using Chrome executable: {chrome_path}")
            launch_kwargs["executable_path"] = chrome_path
        else:
            # Most stable on macOS
            print("[browser] Using Playwright channel=chrome")
            launch_kwargs["channel"] = "chrome"

        self.ctx = await self._pw.chromium.launch_persistent_context(**launch_kwargs)

        self.page = self.ctx.pages[0] if self.ctx.pages else await self.ctx.new_page()

        # Minimal stealth normalization
        await self.ctx.add_init_script("""
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
""")
        await self.ctx.set_extra_http_headers(
            {"Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7"}
        )
        await self._maximize_viewport()

    async def _maximize_viewport(self):
        if not self.page:
            return
        # Best-effort: resize to available screen size (no hardcoded pixels).
        try:
            size = await self.page.evaluate(
                "({width: window.screen.availWidth, height: window.screen.availHeight})"
            )
            width = int(size.get("width") or 0)
            height = int(size.get("height") or 0)
            if width > 0 and height > 0:
                await self.page.set_viewport_size({"width": width, "height": height})
        except Exception:
            pass

    async def stop(self):
        if self.ctx:
            await self.ctx.close()
        if self._pw:
            await self._pw.stop()

    async def goto(self, url: str):
        await self.page.goto(url, wait_until="domcontentloaded")

    async def back(self):
        await self.page.go_back(wait_until="domcontentloaded")

    async def reload(self):
        await self.page.reload(wait_until="domcontentloaded")

    async def wait(self, ms: int = 800):
        await self.page.wait_for_timeout(ms)

    async def scroll(self, dy: int = 700):
        await self.page.mouse.wheel(0, dy)

    async def press(self, key: str):
        await self.page.keyboard.press(key)

    async def click(self, element_id: str, index: int = 0):
        loc = self._locator_for(element_id, index=index)
        await loc.click()

    async def click_at(self, x: int, y: int):
        await self.page.mouse.click(int(x), int(y))

    async def type(self, element_id: str, text: str, clear: bool = True, index: int = 0):
        loc = self._locator_for(element_id, index=index)
        if clear:
            await loc.fill("")
        await loc.type(text, delay=10)

    async def screenshot(self, path: str):
        await self.page.screenshot(path=path, full_page=False)

    async def get_url_title(self) -> dict:
        return {"url": self.page.url, "title": await self.page.title()}

    def _locator_for(self, element_id: str, index: int = 0):
        el = self.element_index[element_id]
        r = el.locator

        # Build locator without strict collisions by always selecting nth(index)
        if r["kind"] == "role":
            loc = self.page.get_by_role(r["role"], name=r.get("name"))
        elif r["kind"] == "label":
            loc = self.page.get_by_label(r["text"])
        elif r["kind"] == "placeholder":
            loc = self.page.get_by_placeholder(r["text"])
        elif r["kind"] == "text":
            loc = self.page.get_by_text(r["text"], exact=r.get("exact", False))
        else:
            loc = self.page.locator(r["css"])

        return loc.nth(int(index))

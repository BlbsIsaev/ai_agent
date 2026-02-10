# src/agent/tools_perception.py
import uuid
from typing import Any, Dict, List, Tuple

INTERACTIVE_SELECTOR = """
a, button, input, textarea, select,
[role="button"], [role="link"], [role="row"], [role="listitem"], [role="option"],
[role="menuitem"], [role="tab"], [role="checkbox"], [role="gridcell"],
[contenteditable="true"]
"""

async def get_visible_text(page, max_chars: int = 2000) -> str:
    return await page.evaluate(
        """
(maxChars) => {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let out = [];
  let total = 0;
  const isHidden = (el) => {
    if (!el) return true;
    if (el.tagName === 'NOSCRIPT') return true;
    if (el.closest && el.closest('noscript')) return true;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return true;
    if (el.getAttribute && el.getAttribute('aria-hidden') === 'true') return true;
    return false;
  };
  while (walker.nextNode()) {
    const parent = walker.currentNode.parentElement;
    if (isHidden(parent)) continue;
    const t = walker.currentNode.nodeValue.trim();
    if (!t) continue;
    const chunk = t.length > 140 ? t.slice(0, 140) : t;
    out.push(chunk);
    total += chunk.length + 1;
    if (total > maxChars) break;
  }
  return out.join(" ");
}
        """,
        int(max_chars),
    )


async def collect_raw_elements(page, limit: int = 260) -> List[Dict[str, Any]]:
    return await page.evaluate(
        f"""
() => {{
  const sel = `{INTERACTIVE_SELECTOR}`;
  const nodes = Array.from(document.querySelectorAll(sel));

  const isVisible = (el) => {{
    const r = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return r.width > 2 && r.height > 2 && style.visibility !== 'hidden' && style.display !== 'none';
  }};

  const pickName = (el) =>
    el.getAttribute('aria-label') ||
    el.getAttribute('data-tooltip') ||
    el.getAttribute('title') ||
    el.innerText?.trim() ||
    el.value?.toString()?.trim() ||
    el.placeholder?.toString()?.trim() ||
    '';

  const clip = (s, n) => (s || '').toString().trim().slice(0, n);

  const findLabel = (el) => {{
    // label[for=id]
    const id = el.id;
    if (id) {{
      const lab = document.querySelector(`label[for="${{CSS.escape(id)}}"]`);
      if (lab && lab.innerText) return lab.innerText.trim();
    }}
    // label wrapping
    const parentLabel = el.closest('label');
    if (parentLabel && parentLabel.innerText) return parentLabel.innerText.trim();
    return null;
  }};

  return nodes.filter(isVisible).slice(0, {int(limit)}).map(el => {{
    const r = el.getBoundingClientRect();
    const tag = el.tagName.toLowerCase();
    const href = (tag === 'a') ? el.href : (el.closest('a') ? el.closest('a').href : null);

    return {{
      tag,
      name: clip(pickName(el), 120),
      role: el.getAttribute('role') || null,
      type: el.getAttribute('type') || null,
      placeholder: el.getAttribute('placeholder') || null,
      label: clip(findLabel(el), 120),
      aria: el.getAttribute('aria-label') || null,
      title: el.getAttribute('title') || null,
      tooltip: el.getAttribute('data-tooltip') || null,
      href: href ? clip(href, 220) : null,
      disabled: !!(el.disabled || el.getAttribute('aria-disabled') === 'true'),
      bbox: [r.x, r.y, r.width, r.height],
      text: clip(el.innerText || '', 180),
    }};
  }});
}}
        """
    )


async def collect_links(page, limit: int = 80) -> List[Dict[str, Any]]:
    return await page.evaluate(
        f"""
() => {{
  const nodes = Array.from(document.querySelectorAll('a'));
  const isVisible = (el) => {{
    const r = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return r.width > 2 && r.height > 2 && style.visibility !== 'hidden' && style.display !== 'none';
  }};

  const pickText = (el) =>
    el.getAttribute('aria-label') ||
    el.getAttribute('title') ||
    el.innerText?.trim() ||
    '';

  const clip = (s, n) => (s || '').toString().trim().slice(0, n);

  return nodes.filter(isVisible).slice(0, {int(limit)}).map(el => {{
    const r = el.getBoundingClientRect();
    return {{
      text: clip(pickText(el), 180),
      href: el.href || null,
      title: el.getAttribute('title') || null,
      bbox: [r.x, r.y, r.width, r.height],
    }};
  }});
}}
        """
    )


async def perceive_page(page) -> Tuple[str, List[Dict[str, Any]]]:
    visible_text = await get_visible_text(page, max_chars=2000)
    raw = await collect_raw_elements(page, limit=260)
    return visible_text, raw


def normalize_elements(raw_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalizes raw DOM element snapshots into a compact, tool-friendly structure.
    Returns universal fields used by the agent:
      id, tag, role, name, text, href, aria, title, disabled, bbox, locator
    """
    norm: List[Dict[str, Any]] = []
    for e in raw_elements:
        eid = f"e{uuid.uuid4().hex[:8]}"
        tag = e.get("tag")
        role = e.get("role")
        name = (e.get("name") or e.get("text") or "").strip()
        text = (e.get("text") or "").strip()

        # Locator recipe (generic, not site-specific)
        # Prefer role-based locators when role + name are present (works for Gmail rows, menus, etc.)
        if e.get("role") and (e.get("name") or e.get("text")):
            locator = {"kind": "role", "role": e["role"], "name": (e.get("name") or e.get("text"))}
        elif tag in ("input", "textarea", "select") and e.get("label"):
            locator = {"kind": "label", "text": e["label"]}
        elif tag in ("input", "textarea", "select") and e.get("placeholder"):
            locator = {"kind": "placeholder", "text": e["placeholder"]}
        elif tag in ("button", "a") and (text or name):
            locator = {"kind": "text", "text": (text or name), "exact": False}
        else:
            # last resort: generic locator by tag
            locator = {"kind": "css", "css": tag or "*"}

        norm.append(
            {
                "id": eid,
                "tag": tag,
                "role": role,
                "name": name,
                "text": text,
                "href": e.get("href"),
                "aria": e.get("aria"),
                "title": e.get("title"),
                "tooltip": e.get("tooltip"),
                "disabled": bool(e.get("disabled")),
                "bbox": e.get("bbox"),
                "locator": locator,
            }
        )

    return norm

Project Context (Agent Architecture + TZ)

Goal
Build an autonomous browser agent that can solve multi-step tasks on real websites with minimal user involvement. The agent must be able to accept a complex task, navigate, act, verify, and finish the task without hardcoded site logic.

Core Architecture
- DOM-first, vision fallback.
- Default loop: observe -> plan (short) -> act (1-2 actions) -> observe (verify).
- Use DOM tools whenever possible; fallback to vision only if DOM is insufficient.

DOM-first Rules
- Always prefer element_id from observe().
- Use click_best_match() for ambiguity.
- Use find_elements() / extract_list() / extract_links() for structured scanning.
- If a click does not change page state, try another candidate or follow href when available.

Vision Fallback Rules
Use vision only when:
- DOM tools cannot locate target (find_elements returns empty or low confidence).
- Repeated DOM clicks do not change the state.
- Visual element exists but is missing from DOM (virtualized lists / SPA issues).
Fallback flow:
1) vision_find(query)
2) click_at(x,y)
3) observe()

Safety / Security
- Destructive actions (delete, pay, confirm, unsubscribe, send, etc.) must go through the security layer for explicit confirmation.
- Do not ask for confirmation in text; execute the risky tool and let the security layer prompt.

No Hardcoding
- No site-specific selectors, URLs, or pre-baked flows.
- No domain-specific scripts (e.g., "Gmail spam cleanup steps").
- The agent must decide actions from current observation.

Context & Output Rules
- Keep responses short and iterative.
- Summary must report only completed actions, not plans.
- Do not finish unless the task is done or user input/confirmation is required.

Models / Delegation (optional)
- Main model: strong reasoning + planning.
- Helper models: cheap extract/summary for DOM or vision.
- Vision model only for fallback.

Tooling Summary
- observe(): compact DOM snapshot
- extract_text(): long text content
- extract_links(): visible links
- extract_list(): list rows/cards
- find_elements(): filtered candidates
- click/type/press/scroll/back/reload
- vision_find()/click_at(): vision fallback

Completion Criteria
The task is complete only when:
- The requested action is done and verified in the UI, or
- The user must manually complete something (login/2FA/captcha), or
- A safety confirmation is required by the security layer.

Original TZ (Condensed, Must-Keep)

Task
Develop an AI agent that autonomously controls a web browser to complete complex multi-step tasks.

Required Functionality
- Browser automation:
  - Programmatic browser control.
  - Persistent sessions (user can log in manually, agent continues).
  - Visible browser (not headless).
- Autonomous AI agent:
  - Uses OpenAI or Claude models.
  - Makes decisions without constant user involvement.
  - Handles multi-step tasks across pages.
- Context management:
  - Cannot send full webpages to the model.
  - Must use strategies to work within token limits.
- Advanced patterns (at least one):
  - Sub-agent architecture.
  - Error handling / recovery.
  - Security layer for destructive actions (pay, delete email, etc.).

Must NOT Include
- Hardcoded action scripts (e.g., “steps to delete spam”).
- Hardcoded selectors or site-specific DOM paths.
- Hardcoded link/element hints (e.g., “vacancies are at /vacancies”).

Free to Choose
- Browser automation library.
- AI SDK/provider.
- Programming language.
- Page data extraction strategy.
- Tool/function calling architecture.
- Handling of dynamic pages, popups, forms.
- MCP usage.

Delivery Expectation
- Record a short demo video: terminal + browser visible.
- In terminal: user enters a short task; agent calls tools with args.
- Agent navigates, clicks, types, and reports results at the end.

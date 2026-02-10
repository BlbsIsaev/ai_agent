from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Memory:
    recent_steps: List[Dict[str, Any]] = field(default_factory=list)
    summaries: List[str] = field(default_factory=list)

    def add_step(self, step: Dict[str, Any], keep_last: int = 10):
        self.recent_steps.append(step)
        if len(self.recent_steps) > keep_last:
            # сворачиваем самый старый шаг в “summary”
            old = self.recent_steps.pop(0)
            self.summaries.append(f"{old.get('tool')}({old.get('args')})->{old.get('result')}")

    def dump_for_prompt(self) -> str:
        s = ""
        if self.summaries:
            s += "Сжатая история:\n- " + "\n- ".join(self.summaries[-20:]) + "\n"
        if self.recent_steps:
            s += "Последние шаги:\n"
            for st in self.recent_steps[-10:]:
                s += f"- {st.get('tool')} {st.get('args')} => {st.get('result')}\n"
        return s.strip()

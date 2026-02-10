import re

DANGEROUS_PATTERNS = [
    r"\bpay\b",
    r"\bcheckout\b",
    r"\bpurchase\b",
    r"\bbuy\b",
    r"\bconfirm\b",
    r"\bsubmit payment\b",
    r"\bdelete\b",
    r"\bremove\b",
    r"\bunsubscribe\b",
    r"\bsend email\b",
    r"\btransfer\b",
    r"\bоплат(ить|а|у|е|ы)?\b",
    r"\bзаказ(ать|а|у|е|ы)?\b",
    r"\bподтверд(ить|и|ите|им|ила|ено|ены)?\b",
    r"\bудал(ить|и|ите|ено|ены)?\b",
    r"\bотпис(аться|ка|ался|ались|ка)?\b",
    r"\bперевод(ить|а|у|е|ы)?\b",
    r"\bперевест(и|у|е|ено|ены)?\b",
]

def is_risky(tool_name: str, args: dict) -> bool:
    s = (tool_name + " " + str(args)).lower()
    return any(re.search(p, s) for p in DANGEROUS_PATTERNS)

def ask_user_confirmation(tool_name: str, args: dict) -> bool:
    print(f"\n⚠️ Рискованное действие: {tool_name} {args}")
    ans = input("Разрешить? (y/N): ").strip().lower()
    return ans == "y"

import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console

from .orchestrator import AgentOrchestrator

console = Console()

async def amain():
    load_dotenv()
    orch = AgentOrchestrator(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        cheap_model=os.getenv("OPENAI_CHEAP_MODEL", "gpt-4.1-nano"),
        user_data_dir=os.getenv("USER_DATA_DIR", ".user_data"),
        slow_mo_ms=int(os.getenv("SLOW_MO_MS", "120")),
    )
    await orch.start()

    console.print("[bold green]Browser agent started.[/bold green]")
    console.print("Введите задачу (или пусто чтобы выйти).\n")

    try:
        while True:
            task = input("> ").strip()
            if not task:
                break
            await orch.run_task(task)
            console.print("\n[bold cyan]Готово. Можешь дать следующую задачу.[/bold cyan]\n")
    finally:
        await orch.stop()

def main():
    asyncio.run(amain())

if __name__ == "__main__":
    main()

import asyncio
from rich.console import Console
from rich.table import Table

from browser import BrowserController


console = Console()


def render_candidates(cands, max_rows: int = 25):
    table = Table(title=f"Candidates (showing up to {max_rows})", show_lines=False)
    table.add_column("id", style="bold")
    table.add_column("role")
    table.add_column("name")
    table.add_column("nth", justify="right")
    table.add_column("disabled", justify="center")
    table.add_column("checked", justify="center")
    table.add_column("expanded", justify="center")

    for c in cands[:max_rows]:
        table.add_row(
            c.id,
            c.role,
            (c.name or "")[:80],
            str(c.nth),
            "" if c.disabled is None else str(c.disabled),
            "" if c.checked is None else str(c.checked),
            "" if c.expanded is None else str(c.expanded),
        )
    return table


async def repl():
    bc = BrowserController(headless=False)
    await bc.start()
    console.print("[green]Browser started (headful). Type 'help' for commands.[/green]")

    try:
        while True:
            cmd = console.input("[bold cyan]agent> [/bold cyan]").strip()
            if not cmd:
                continue

            if cmd in ("q", "quit", "exit"):
                break

            if cmd == "help":
                console.print(
                    "\n[bold]Commands:[/bold]\n"
                    "  goto <url>        - open url\n"
                    "  obs               - observe (aria snapshot + candidates)\n"
                    "  snap              - print aria snapshot (last obs)\n"
                    "  quit              - exit\n"
                )
                continue

            if cmd.startswith("goto "):
                url = cmd[5:].strip()
                console.print(f"[yellow]Navigating to:[/yellow] {url}")
                try:
                    await bc.goto(url)
                    console.print("[green]Loaded.[/green]")
                except Exception as e:
                    console.print(f"[red]goto error:[/red] {type(e).__name__}: {e}")
                continue

            if cmd == "obs":
                state = await bc.observe(max_candidates=40)
                console.print(f"[bold]URL:[/bold] {state.url}")
                console.print(f"[bold]Title:[/bold] {state.title}")
                console.print(render_candidates(state.candidates, max_rows=25))
                # store snapshot on controller as last state in-memory:
                bc._last_snapshot = state.aria_snapshot  # quick & dirty for MVP
                continue

            if cmd == "snap":
                snap = getattr(bc, "_last_snapshot", None)
                if not snap:
                    console.print("[red]No snapshot yet. Run 'obs' first.[/red]")
                else:
                    console.print(snap)
                continue

            console.print("[red]Unknown command.[/red] Type 'help'.")


            if cmd.startswith("click "):
                cid = cmd.split(maxsplit=1)[1].strip()
                console.print(await bc.click(cid))
                continue

            if cmd.startswith("type "):
                parts = cmd.split(maxsplit=2)
                if len(parts) < 3:
                    console.print("[red]Usage: type <id> <text>[/red]")
                    continue
                _, cid, text = parts
                console.print(await bc.type(cid, text))
                continue

            if cmd.startswith("select "):
                parts = cmd.split(maxsplit=2)
                if len(parts) < 3:
                    console.print("[red]Usage: select <id> <option_text>[/red]")
                    continue
                _, cid, opt = parts
                console.print(await bc.select(cid, opt))
                continue

            if cmd.startswith("scroll"):
                parts = cmd.split()
                direction = parts[1] if len(parts) > 1 else "down"
                amount = int(parts[2]) if len(parts) > 2 else 800
                console.print(await bc.scroll(direction=direction, amount=amount))
                continue

            if cmd == "back":
                console.print(await bc.back())
                continue


    finally:
        await bc.close()
        console.print("[green]Browser closed.[/green]")





def main():
    asyncio.run(repl())


if __name__ == "__main__":
    main()

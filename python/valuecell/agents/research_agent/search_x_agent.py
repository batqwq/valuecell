"""Periodic Grok search for latest cryptocurrency market updates."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiofiles
from loguru import logger

from valuecell.agents.research_agent.knowledge import insert_md_file_to_knowledge
from valuecell.integrations.xai_client import grok_search
from valuecell.utils.path import get_knowledge_path

DEFAULT_INTERVAL_MINUTES = 10
DEFAULT_PROMPT = (
    "Search X (Twitter) and the broader internet for the latest cryptocurrency market "
    "developments. Summarize price action, notable tweets, regulatory headlines, "
    "institutional flows, macro drivers, and large on-chain transactions. Include timestamps, "
    "source handles or URLs, and highlight potential impacts for the next 24 hours."
)


async def fetch_market_update(prompt: str) -> str:
    """Call Grok to retrieve a market update summary."""

    system_prompt = (
        "You are a crypto market analyst acting strictly as an information search engine. "
        "Gather breaking updates from X (Twitter) and trusted web sources, cross-check for credibility, "
        "deduplicate overlapping items, and always cite sources. Do NOT provide any investment advice or "
        "recommendations; only report verified, factual updates and their context."
    )

    extra_body = {
        # These hints are optional and future-proofed for xAI API extensions.
        "search": {"channels": ["x", "web"]},
    }

    summary = await grok_search(
        prompt,
        system_prompt=system_prompt,
        temperature=0.2,
        max_output_tokens=2_000,
        extra_body=extra_body,
    )
    return summary


async def write_update_to_knowledge(content: str) -> Path:
    """Persist the content to the knowledge store and index it."""

    knowledge_root = Path(get_knowledge_path()) / "market_updates"
    knowledge_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    file_name = f"crypto_market_update_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
    path = knowledge_root / file_name

    header = (
        f"# Cryptocurrency Market Update ({timestamp.strftime('%Y-%m-%d %H:%M UTC')})\n\n"
    )
    async with aiofiles.open(path, "w", encoding="utf-8") as file:
        await file.write(header + content.strip() + "\n")

    metadata = {
        "source": "grok-4-fast",
        "topic": "crypto_market",
        "timestamp": timestamp.isoformat(),
    }
    await insert_md_file_to_knowledge(
        name=file_name,
        path=path,
        metadata=metadata,
    )
    logger.info("Stored market update at %s", path)
    return path


async def run_once(prompt: str) -> None:
    summary = await fetch_market_update(prompt)
    await write_update_to_knowledge(summary)


async def run_periodically(interval_minutes: int, prompt: str) -> None:
    """Run the fetch task every `interval_minutes` minutes."""
    interval_seconds = max(interval_minutes, 1) * 60
    logger.info(
        "Starting Grok market watcher (interval=%s minutes)", interval_minutes
    )
    while True:
        try:
            await run_once(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception("searchXagent iteration failed: %s", exc)
        await asyncio.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Periodic Grok search for latest cryptocurrency market updates."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.getenv("SEARCH_X_AGENT_INTERVAL_MINUTES", DEFAULT_INTERVAL_MINUTES)),
        help="Interval in minutes between updates (default: %(default)s).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single update instead of looping forever.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=os.getenv("SEARCH_X_AGENT_PROMPT", DEFAULT_PROMPT),
        help="Custom prompt to send to Grok.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    if args.once:
        await run_once(args.prompt)
    else:
        await run_periodically(args.interval, args.prompt)


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("searchXagent stopped by user.")


if __name__ == "__main__":
    main()

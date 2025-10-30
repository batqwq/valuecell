"""Long-polling Telegram bridge.

Fetches updates via getUpdates and forwards them to the local webhook endpoint
so the same processing pipeline handles both webhook and polling.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests
from requests import exceptions as req_exc


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None else default


def main():
    token = env("TELEGRAM_BOT_TOKEN")
    if not token:
        print("TELEGRAM_BOT_TOKEN not set")
        return

    api_host = env("API_HOST", "127.0.0.1")
    api_port = env("API_PORT", "8000")
    webhook_secret = env("TELEGRAM_WEBHOOK_SECRET", "")

    webhook_url = f"http://{api_host}:{api_port}/api/v1/telegram/webhook"
    tg_api = f"https://api.telegram.org/bot{token}"

    offset = None
    session = requests.Session()

    print("Starting Telegram long-polling...")
    while True:
        try:
            params = {"timeout": 50}
            if offset is not None:
                params["offset"] = offset
            r = session.get(f"{tg_api}/getUpdates", params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                time.sleep(2)
                continue

            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                headers = {}
                if webhook_secret:
                    headers["X-Telegram-Bot-Api-Secret-Token"] = webhook_secret
                try:
                    session.post(
                        webhook_url,
                        json=upd,
                        headers=headers,
                        timeout=(5, 120),
                    )
                except req_exc.ReadTimeout:
                    print("Webhook 调用读取超时，继续轮询...", flush=True)
                    continue

        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print("Polling error:", e)
            time.sleep(3)


if __name__ == "__main__":
    main()

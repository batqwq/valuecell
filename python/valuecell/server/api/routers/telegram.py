"""Telegram webhook router.

Exposes POST /telegram/webhook to receive updates from Telegram servers.
Validates optional secret header and dispatches to TelegramService.
"""

from fastapi import APIRouter, Header, HTTPException, Request

from valuecell.server.services.telegram_service import TelegramService


def create_telegram_router() -> APIRouter:
    router = APIRouter(prefix="/telegram", tags=["Telegram"])
    service = TelegramService()

    @router.post("/webhook")
    async def telegram_webhook(
        request: Request,
        x_telegram_bot_api_secret_token: str | None = Header(default=None),
    ):
        if not service.is_ready():
            raise HTTPException(status_code=503, detail="Telegram not configured")

        if not service.validate_secret(x_telegram_bot_api_secret_token):
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        await service.handle_update(payload)
        return {"ok": True}

    return router


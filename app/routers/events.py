"""The tracking endpoint. Everything here is on the browser's critical path, so it does
the least work that is correct: validate, stamp, enqueue, return 202."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import ANON_COOKIE, get_current_user, new_anon_id
from app.models import User
from app.ratelimit import events_limiter
from app.schemas import EventBatchAccepted, EventBatchIn, EventIn
from app.security import cookie_kwargs
from app.services.events import ingestor, to_row

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/events", tags=["events"])

ANON_COOKIE_MAX_AGE = 60 * 60 * 24 * 365


@router.post("/batch", status_code=status.HTTP_202_ACCEPTED, response_model=EventBatchAccepted)
async def ingest_batch(
    payload: EventBatchIn,
    request: Request,
    response: Response,
    user: User | None = Depends(get_current_user),
    _db: Session = Depends(get_db),
) -> EventBatchAccepted:
    anon_id = request.cookies.get(ANON_COOKIE) or payload.anon_id or new_anon_id()

    limiter_key = f"u{user.id}" if user else f"a{anon_id or request.client.host}"
    if not events_limiter.allow(limiter_key, cost=len(payload.events)):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "slow down")

    rows, rejected = [], 0
    for raw in payload.events:
        try:
            event = EventIn.model_validate(raw)
        except ValidationError:
            rejected += 1
            continue
        rows.append(
            to_row(event, user_id=user.id if user else None, anon_id=anon_id, path_fallback="")
        )
    if rejected:
        log.warning("dropped invalid events", extra={"rejected": rejected, "kept": len(rows)})

    accepted = ingestor.submit(rows)

    if request.cookies.get(ANON_COOKIE) != anon_id:
        response.set_cookie(ANON_COOKIE, anon_id, **cookie_kwargs(ANON_COOKIE_MAX_AGE))

    return EventBatchAccepted(accepted=accepted, rejected=rejected)

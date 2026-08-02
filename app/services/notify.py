"""Email delivery for the scheduled digest.

Three backends, chosen by configuration rather than by code changes:

  * **SMTP** when SMTP_HOST is set — the real path.
  * **File sink** when it is not. Renders the exact same HTML to ./data/outbox/ so the
    scheduler is demonstrably working without anyone handing out mail credentials.
    This is what makes the feature reviewable by someone who just cloned the repo.
  * **SES** when MAIL_BACKEND=ses, for the AWS deployment path.

Sending is idempotent at the database level, not here: every send is guarded by a
Notification row with a unique `digest:<user>:<date>` key, so a duplicate scheduler
tick or a retried job cannot email anyone twice.
"""

import logging
import smtplib
import ssl
from datetime import date
from email.message import EmailMessage

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import ROOT, settings
from app.models import Notification, User

log = logging.getLogger(__name__)

OUTBOX_DIR = ROOT / "data" / "outbox"


class SmtpNotifier:
    backend = "smtp"

    def send(self, to: str, subject: str, html: str, text: str) -> str:
        message = EmailMessage()
        message["From"] = settings.mail_from
        message["To"] = to
        message["Subject"] = subject
        message.set_content(text)
        message.add_alternative(html, subtype="html")

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
            if settings.smtp_starttls:
                server.starttls(context=ssl.create_default_context())
            if settings.smtp_user:
                server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(message)
        return f"smtp:{settings.smtp_host}"


class FileNotifier:
    backend = "file"

    def send(self, to: str, subject: str, html: str, text: str) -> str:
        OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
        safe = to.replace("@", "_at_").replace("/", "_")
        path = OUTBOX_DIR / f"{date.today().isoformat()}-{safe}.html"
        path.write_text(
            f"<!-- To: {to}\n     Subject: {subject} -->\n{html}", encoding="utf-8"
        )
        log.info("digest written to file sink", extra={"path": str(path)})
        return str(path)


class SesNotifier:
    backend = "ses"

    def send(self, to: str, subject: str, html: str, text: str) -> str:
        import boto3

        client = boto3.client("ses", region_name=settings.aws_region)
        client.send_email(
            Source=settings.mail_from,
            Destination={"ToAddresses": [to]},
            Message={
                "Subject": {"Data": subject},
                "Body": {"Text": {"Data": text}, "Html": {"Data": html}},
            },
        )
        return "ses"


def get_notifier():
    if settings.mail_backend == "ses":
        return SesNotifier()
    if settings.has_smtp:
        return SmtpNotifier()
    return FileNotifier()


def send_once(
    db: Session,
    user: User,
    *,
    dedupe_key: str,
    subject: str,
    html: str,
    text: str,
    recommendation_id: int | None = None,
) -> bool:
    """Send exactly once. Returns True if this call actually delivered.

    The Notification row is written *before* the send and committed, so two concurrent
    workers race on the unique index rather than both mailing the user. Losing that
    race is a normal outcome, not an error.
    """
    notification = Notification(
        user_id=user.id,
        recommendation_id=recommendation_id,
        channel=get_notifier().backend,
        subject=subject[:200],
        status="sending",
        dedupe_key=dedupe_key,
    )
    db.add(notification)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        log.info("digest already sent", extra={"user_id": user.id, "key": dedupe_key})
        return False

    try:
        destination = get_notifier().send(user.email, subject, html, text)
        notification.status = "sent"
        notification.error = destination[:400]
    except Exception as exc:
        # Keep the row so the failure is visible and the retry is deliberate rather
        # than an accidental duplicate on the next tick.
        notification.status = "failed"
        notification.error = str(exc)[:400]
        db.commit()
        log.exception("digest send failed", extra={"user_id": user.id})
        return False

    db.commit()
    log.info("digest sent", extra={"user_id": user.id, "channel": notification.channel})
    return True

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from typing import List

logger = logging.getLogger(__name__)


def send_email(to_addrs: List[str], subject: str, body: str, smtp_server: str, smtp_port: int,
               smtp_user: str, smtp_password: str, from_addr: str = None) -> None:
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = from_addr or smtp_user
    msg['To'] = ", ".join(to_addrs)
    msg.attach(MIMEText(body, 'plain'))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.sendmail(msg['From'], to_addrs, msg.as_string())
        logger.info("Email sent to %s", to_addrs)
    except Exception as e:
        logger.warning("Failed to send email: %s", e)


import smtplib
from email.mime.text import MIMEText
from typing import Optional


def send_email(subject: str, body: str, to_addr: str, from_addr: str, smtp_server: str,
               smtp_port: int, smtp_user: str, smtp_password: str) -> Optional[str]:
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return None
    except Exception as e:
        return str(e)


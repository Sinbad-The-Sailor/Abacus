# -*- coding: utf-8 -*-
import os
import smtplib
import datetime as dt

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_email(msg: str, status: str):

    host = os.getenv("EMAIL_HOST")
    port = int(os.getenv("EMAIL_PORT"))
    adrs = os.getenv("EMAIL_ADRS")
    pasw = os.getenv("EMAIL_PASW")

    recipient = adrs
    recipient = "karlaxel.n@outlook.com"
    sender = adrs

    message = _build_full_msg(msg, status)

    with smtplib.SMTP(host=host, port=port) as server:
        server.starttls()
        server.login(user=adrs,
                     password=pasw)
        server.sendmail(sender, recipient, message.as_string())
        server.close()


def _build_full_msg(msg: str, status: str) -> MIMEMultipart:
    header = _build_header(status)
    log = _build_log()
    msg = f"{header}\n\n{msg}\n\n{log}"
    msg = msg.replace('\n', '<br>')
    msg = "<pre><code>" + msg + "</code></pre>"

    message = MIMEMultipart()
    message['Subject'] = f"Investment Report {status}"
    html = MIMEText(msg, "html")
    message.attach(html)

    return message


def _build_header(status: str) -> str:
    file = open("src/abacus/utilities/email_header.txt", "r")
    logo_offset = " " * 11
    logo = ""
    for line in file:
        logo += logo_offset + line.rstrip("\n") + "\n"
    file.close()
    date = str(dt.date.today())
    date_offset = "-" * 32
    if status == "OK":
        stat_offset = "-" * 32
    elif status == "CRITICAL":
        stat_offset = "-" * 29
    else:
        raise ValueError("Invalid status for email header.")

    header = f"{logo}\n\n{date_offset} {date} {date_offset}\n{stat_offset} STATUS: {status} {stat_offset}"
    return header


def _build_log() -> str:
    file = open(".log", "r")
    log = file.read()
    file.close()
    offset = "-" * 35
    log = f"{offset} .LOG {offset}\n{log}"
    return log

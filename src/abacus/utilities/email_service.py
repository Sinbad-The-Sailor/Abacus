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
    msg = f"{header}\n\n {msg}"
    msg = msg.replace('\n', '<br>')
    msg = "<pre><code>" + msg + "</code></pre>"

    message = MIMEMultipart()
    message['Subject'] = f"Investment Report {status}"
    html = MIMEText(msg, "html")
    message.attach(html)

    return message


def _build_header(status: str) -> str:
    file = open("src/abacus/utilities/email_header.txt", "r")
    logo = file.read()
    date = str(dt.date.today())
    date_offset = "-" * 21

    if status == "OK":
        stat_offset = "-" * 21
    elif status == "CRITICAL":
        stat_offset = "-" * 18
    else:
        raise ValueError("Invalid status for email header.")

    header = f"{logo}\n\n {date_offset} {date} {date_offset}\n {stat_offset} STATUS: {status} {stat_offset}"
    file.close()
    return header

# -*- coding: utf-8 -*-
import os
import smtplib
import datetime as dt


def send_email(msg: str):
    host = os.getenv("EMAIL_HOST")
    port = int(os.getenv("EMAIL_PORT"))
    adrs = str(os.getenv("EMAIL_ADRS"))
    pasw = str(os.getenv("EMAIL_PASW"))

    print(adrs, pasw)

    with smtplib.SMTP(host=host, port=port) as server:
        server.starttls()
        server.login(user=adrs,
                     password=pasw)
        server.send_message(msg=str(msg), from_addr=adrs, to_addrs=adrs)
        server.close()


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


class Emailer:
    pass

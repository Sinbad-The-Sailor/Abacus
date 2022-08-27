# -*- coding: utf-8 -*-
import os
import smtplib
import datetime as dt

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class EmailService:
    """
    EmailService to send formatted email as a status update once program is finished.
    """

    def __init__(self, msg, status):
        self.msg = msg
        self.status = status

    def send_email(self):
        """
        Connects to email service and sends a message.
        """
        host = os.getenv("EMAIL_HOST")
        port = int(os.getenv("EMAIL_PORT"))
        adrs = os.getenv("EMAIL_ADRS")
        pasw = os.getenv("EMAIL_PASW")

        recipient = adrs
        sender = adrs

        message = self._build_full_msg()

        with smtplib.SMTP(host=host, port=port) as server:
            server.starttls()
            server.login(user=adrs, password=pasw)
            server.sendmail(sender, recipient, message)
            server.close()

    def _build_full_msg(self) -> str:
        """
        Creates full email message with HTML formatting.

        Returns:
            str: HTML formatted message.
        """
        header = self._build_header()
        log = self._build_log()
        msg = f"{header}\n\n{self.msg}\n\n{log}"
        msg = msg.replace("\n", "<br>")
        msg = "<pre><code>" + msg + "</code></pre>"

        message = MIMEMultipart()
        message["Subject"] = f"Investment Report {self.status}"
        html = MIMEText(msg, "html")
        message.attach(html)

        return message.as_string()

    def _build_header(self) -> str:
        """
        Creates the email header including a status code and date.

        Raises:
            ValueError: Invalid status code.

        Returns:
            str: Formatted header for email.
        """
        file = open("src/abacus/utilities/email_header.txt", "r")
        logo_offset = " " * 11
        logo = ""
        for line in file:
            logo += logo_offset + line.rstrip("\n") + "\n"
        file.close()
        date = str(dt.date.today())
        date_offset = "-" * 32
        if self.status == "OK":
            stat_offset = "-" * 32
        elif self.status == "CRITICAL":
            stat_offset = "-" * 29
        else:
            raise ValueError("Invalid status for email header.")

        header = f"{logo}\n\n{date_offset} {date} {date_offset}\n{stat_offset} STATUS: {self.status} {stat_offset}"

        return header

    @staticmethod
    def _build_log() -> str:
        """
        Reads the .log file and creates a formatted string.

        Returns:
            str: Formatted log for email.
        """
        file = open(".log", "r")
        log = file.read()
        file.close()
        offset = "-" * 35
        log = f"{offset} .LOG {offset}\n{log}"

        return log

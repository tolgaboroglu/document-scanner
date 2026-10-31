"""Export a contact as vCard (for phones and address books), JSON or CSV."""

from __future__ import annotations

import csv
import io
import json

from .extract import Contact


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", "\\n")


def to_vcard(contact: Contact) -> str:
    name = (contact.values("NAME") or [""])[0]
    parts = name.split()
    family = parts[-1] if len(parts) > 1 else ""
    given = " ".join(parts[:-1]) if len(parts) > 1 else name
    lines = ["BEGIN:VCARD", "VERSION:3.0", f"N:{_esc(family)};{_esc(given)};;;", f"FN:{_esc(name or 'Unknown')}"]
    for org in contact.values("ORG")[:1]:
        lines.append(f"ORG:{_esc(org)}")
    for title in contact.values("DES")[:1]:
        lines.append(f"TITLE:{_esc(title)}")
    for phone in contact.values("PHONE"):
        lines.append(f"TEL;TYPE=WORK,VOICE:{phone}")
    for email in contact.values("EMAIL"):
        lines.append(f"EMAIL;TYPE=WORK:{email}")
    for web in contact.values("WEB"):
        lines.append(f"URL:{web if web.startswith('http') else 'https://' + web}")
    lines.append("END:VCARD")
    return "\r\n".join(lines) + "\r\n"


def to_json(contact: Contact) -> str:
    return json.dumps(contact.to_dict()["grouped"], ensure_ascii=False, indent=2)


def to_csv(contact: Contact) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["label", "value", "source"])
    for f in contact.fields:
        writer.writerow([f.label, f.value, f.source])
    return buf.getvalue()

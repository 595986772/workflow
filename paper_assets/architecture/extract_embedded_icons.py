#!/usr/bin/env python3
"""Extract reusable image assets embedded in the canonical Draw.io source."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "hero_dag_system_model_polished_v8_scheduler_gap_plus30_editable.drawio"
OUTPUT = ROOT / "icons"


def asset_name(cell_id: str, extension: str) -> str:
    groups = (
        ("device-phone", "device_phone"),
        ("device-laptop", "device_laptop"),
        ("device-vehicle", "device_vehicle"),
        ("service-repository", "service_repository"),
        ("local-scheduler", "local_scheduler"),
        ("legend-local-scheduler", "local_scheduler"),
        ("server-", "edge_server"),
    )
    for prefix, name in groups:
        if cell_id.startswith(prefix):
            return f"{name}.{extension}"
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", cell_id)
    return f"{safe_id}.{extension}"


def decode_data_uri(uri: str) -> tuple[str, bytes]:
    header, payload = uri.split(",", 1)
    mime = header[5:].split(";", 1)[0]
    if ";base64" in header:
        return mime, base64.b64decode(payload)
    return mime, urllib.parse.unquote_to_bytes(payload)


def main() -> None:
    root = ET.parse(SOURCE).getroot()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assets: dict[str, dict[str, object]] = {}

    for cell in root.iter("mxCell"):
        style = cell.get("style", "")
        match = re.search(r"(?:^|;)image=([^;]+)", style)
        if not match or not match.group(1).startswith("data:image"):
            continue

        cell_id = cell.get("id", "unnamed")
        mime, content = decode_data_uri(urllib.parse.unquote(match.group(1)))
        extension = {"image/svg+xml": "svg", "image/png": "png"}.get(mime)
        if extension is None:
            raise ValueError(f"Unsupported embedded MIME type: {mime}")

        filename = asset_name(cell_id, extension)
        digest = hashlib.sha256(content).hexdigest()
        existing = assets.get(filename)
        if existing and existing["sha256"] != digest:
            raise ValueError(f"Conflicting embedded assets mapped to {filename}")

        if not existing:
            (OUTPUT / filename).write_bytes(content)
            assets[filename] = {
                "file": filename,
                "mime_type": mime,
                "sha256": digest,
                "occurrences": [],
            }
        assets[filename]["occurrences"].append(cell_id)

    manifest = {
        "source": SOURCE.name,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "embedded_object_count": sum(len(item["occurrences"]) for item in assets.values()),
        "unique_asset_count": len(assets),
        "assets": list(assets.values()),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Extracted {manifest['embedded_object_count']} objects "
        f"into {manifest['unique_asset_count']} unique assets."
    )


if __name__ == "__main__":
    main()

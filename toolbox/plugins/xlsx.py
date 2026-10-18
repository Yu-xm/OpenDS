"""Bounded coordinate-preserving reader for XLSX workbooks."""

from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from zipfile import BadZipFile, ZipFile

from .base import FileReader, MessageContent


XLSX_TEXT_LIMIT = 48_000
CELL_REFERENCE = re.compile(r"([A-Z]+)(\d+)")
RELATIONSHIP_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _rich_text(element: ET.Element) -> str:
    return "".join(
        child.text or "" for child in element.iter() if _local_name(child.tag) == "t"
    )


def _shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    values: list[str] = []
    with archive.open("xl/sharedStrings.xml") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if _local_name(element.tag) == "si":
                values.append(_rich_text(element))
                element.clear()
    return values


def _sheets(archive: ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        relation.attrib.get("Id", ""): relation.attrib.get("Target", "")
        for relation in relationships
        if _local_name(relation.tag) == "Relationship"
    }
    sheets = []
    for element in workbook.iter():
        if _local_name(element.tag) != "sheet":
            continue
        target = targets.get(element.attrib.get(RELATIONSHIP_ID, ""), "")
        if target:
            path = posixpath.normpath(posixpath.join("xl", target.lstrip("/")))
            sheets.append((element.attrib.get("name", "Sheet"), path))
    return sheets


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    value = next(
        (child.text or "" for child in cell if _local_name(child.tag) == "v"),
        "",
    )
    if cell_type == "s" and value.isdigit():
        index = int(value)
        return shared_strings[index] if index < len(shared_strings) else value
    if cell_type == "inlineStr":
        return _rich_text(cell)
    if cell_type == "b":
        return "TRUE" if value == "1" else "FALSE"
    formula = next(
        (child.text or "" for child in cell if _local_name(child.tag) == "f"),
        "",
    )
    return value or (f"={formula}" if formula else "")


class XlsxReader(FileReader):
    suffixes = (".xlsx",)

    def read(
        self, filename: str, content: bytes, mime_type: str
    ) -> MessageContent:
        del mime_type
        try:
            archive = ZipFile(BytesIO(content))
        except BadZipFile as error:
            raise ValueError("download is not a valid XLSX workbook") from error
        with archive:
            required = {"xl/workbook.xml", "xl/_rels/workbook.xml.rels"}
            if not required.issubset(archive.namelist()):
                raise ValueError("XLSX workbook metadata is incomplete")
            shared_strings = _shared_strings(archive)
            sheets = _sheets(archive)
            output = [
                f"Workbook: {filename}",
                f"Size: {len(content)} bytes",
                f"Sheets: {len(sheets)} ({', '.join(name for name, _ in sheets)})",
            ]
            characters = sum(len(line) + 1 for line in output)
            truncated = False
            for sheet_name, sheet_path in sheets:
                if sheet_path not in archive.namelist():
                    output.append(f"\n## Sheet: {sheet_name}\n[worksheet XML missing]")
                    continue
                output.append(f"\n## Sheet: {sheet_name}")
                with archive.open(sheet_path) as stream:
                    for _, row in ET.iterparse(stream, events=("end",)):
                        if _local_name(row.tag) != "row":
                            continue
                        values = []
                        for cell in row:
                            if _local_name(cell.tag) != "c":
                                continue
                            reference = cell.attrib.get("r", "?")
                            value = " ".join(_cell_value(cell, shared_strings).split())
                            if value:
                                values.append(f"{reference}={value[:500]}")
                        if values:
                            reference = values[0].split("=", 1)[0]
                            match = CELL_REFERENCE.fullmatch(reference)
                            row_number = row.attrib.get("r") or (
                                match.group(2) if match else "?"
                            )
                            line = f"Row {row_number}: " + " | ".join(values)
                            if characters + len(line) + 1 > XLSX_TEXT_LIMIT:
                                truncated = True
                                row.clear()
                                break
                            output.append(line)
                            characters += len(line) + 1
                        row.clear()
                if truncated:
                    break
            if truncated:
                output.append(
                    f"\n[Workbook text truncated at {XLSX_TEXT_LIMIT} characters.]"
                )
            return [{"type": "text", "text": "\n".join(output)}]

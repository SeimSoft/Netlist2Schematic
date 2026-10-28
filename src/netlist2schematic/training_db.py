"""SQLite store for manually approved placements ("Send training data")."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
create table if not exists examples (
    id integer primary key autoincrement,
    circuit text not null,
    netlist text not null,
    placement text not null,
    source text not null,
    created_at text not null,
    pin_maps text not null default '{}'
)
"""


@dataclass(frozen=True)
class Example:
    id: int
    circuit: str
    netlist: str
    placement: dict[str, tuple[int, int, int]]
    source: str
    created_at: str
    pin_maps: dict[str, dict[str, int]]


class TrainingDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(SCHEMA)
            columns = {row[1] for row in con.execute("pragma table_info(examples)")}
            if "pin_maps" not in columns:
                con.execute("alter table examples add column pin_maps text not null default '{}'")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add(
        self,
        circuit: str,
        netlist: str,
        placement: dict[str, tuple[int, int, int]],
        source: str = "manual",
        pin_maps: dict[str, dict[str, int]] | None = None,
    ) -> int:
        payload = json.dumps({name: list(pos) for name, pos in sorted(placement.items())})
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as con:
            cur = con.execute(
                "insert into examples (circuit, netlist, placement, source, created_at, pin_maps) values (?, ?, ?, ?, ?, ?)",
                (circuit, netlist, payload, source, stamp, json.dumps(pin_maps or {})),
            )
            return int(cur.lastrowid)

    def examples(self, circuit: str | None = None) -> list[Example]:
        query = "select id, circuit, netlist, placement, source, created_at, pin_maps from examples"
        args: tuple = ()
        if circuit is not None:
            query += " where circuit = ?"
            args = (circuit,)
        query += " order by id"
        with self._connect() as con:
            rows = con.execute(query, args).fetchall()
        return [
            Example(
                id=row[0],
                circuit=row[1],
                netlist=row[2],
                placement={name: tuple(pos) for name, pos in json.loads(row[3]).items()},
                source=row[4],
                created_at=row[5],
                pin_maps=json.loads(row[6] or "{}"),
            )
            for row in rows
        ]

    def delete(self, ids: list[int]) -> int:
        """Remove examples by id; returns how many were deleted."""
        with self._connect() as con:
            cur = con.executemany("delete from examples where id = ?", [(i,) for i in ids])
            return cur.rowcount

    def count(self) -> int:
        with self._connect() as con:
            return int(con.execute("select count(*) from examples").fetchone()[0])

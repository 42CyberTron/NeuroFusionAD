"""Local-only SQLite storage for explicitly entered demonstration/research records."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PatientStore:
    def __init__(self, path: str | Path = "database/neurofusion.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS visits (
                    visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
                    visit_date TEXT NOT NULL,
                    clinical_json TEXT NOT NULL,
                    mri_path TEXT,
                    pet_path TEXT,
                    UNIQUE(patient_id, visit_date)
                );
                CREATE INDEX IF NOT EXISTS visits_patient_date
                    ON visits(patient_id, visit_date);
                """
            )

    def add_patient(self, patient_id: str, display_name: str) -> None:
        patient_id = patient_id.strip()
        display_name = display_name.strip()
        if not patient_id or not display_name:
            raise ValueError("Patient ID and display name are required.")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO patients(patient_id, display_name, created_at) VALUES (?, ?, ?)",
                (patient_id, display_name, datetime.now(timezone.utc).isoformat()),
            )

    def add_visit(
        self,
        patient_id: str,
        visit_date: str,
        clinical: dict[str, Any],
        mri_path: str | None = None,
        pet_path: str | None = None,
    ) -> None:
        try:
            datetime.fromisoformat(visit_date)
        except ValueError as exc:
            raise ValueError("Visit date must be a valid ISO date.") from exc
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO visits(patient_id, visit_date, clinical_json, mri_path, pet_path)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    patient_id,
                    visit_date,
                    json.dumps(clinical, allow_nan=False),
                    mri_path,
                    pet_path,
                ),
            )

    def list_patients(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT p.patient_id, p.display_name, COUNT(v.visit_id) AS visit_count
                   FROM patients p LEFT JOIN visits v ON v.patient_id = p.patient_id
                   GROUP BY p.patient_id ORDER BY p.display_name"""
            ).fetchall()
        return [dict(row) for row in rows]

    def get_visits(self, patient_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT visit_date, clinical_json, mri_path, pet_path
                   FROM visits WHERE patient_id = ? ORDER BY visit_date""",
                (patient_id,),
            ).fetchall()
        return [
            {
                "visit_date": row["visit_date"],
                "clinical": json.loads(row["clinical_json"]),
                "mri_path": row["mri_path"],
                "pet_path": row["pet_path"],
            }
            for row in rows
        ]

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT patient_id, display_name FROM patients WHERE patient_id = ?",
                (patient_id,),
            ).fetchone()
        return dict(row) if row else None

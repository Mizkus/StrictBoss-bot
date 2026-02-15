import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from dateutil.rrule import rrulestr
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

UPLOAD_CALENDAR = 1


@dataclass(frozen=True)
class Config:
    token: str
    allowed_user_ids: tuple[int, ...]
    admin_user_ids: tuple[int, ...]
    timezone: ZoneInfo
    db_path: Path
    default_fine: int = 100


def parse_int_list(env_value: str) -> tuple[int, ...]:
    ids = []
    for part in env_value.split(","):
        value = part.strip()
        if not value:
            continue
        ids.append(int(value))
    return tuple(ids)


def load_config() -> Config:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("BOT_TOKEN is required")

    allowed_raw = os.getenv("ALLOWED_USER_IDS", "").strip()
    allowed = parse_int_list(allowed_raw)
    if len(allowed) != 2:
        raise RuntimeError("ALLOWED_USER_IDS must contain exactly 2 telegram user ids")

    admin_raw = os.getenv("ADMIN_USER_IDS", "").strip()
    admins = parse_int_list(admin_raw) if admin_raw else allowed

    tz_name = os.getenv("TZ", "Europe/Moscow").strip()
    timezone = ZoneInfo(tz_name)
    db_path = Path(os.getenv("DB_PATH", "bot.sqlite3"))
    default_fine = int(os.getenv("DEFAULT_FINE_AMOUNT", "100"))

    return Config(
        token=token,
        allowed_user_ids=allowed,
        admin_user_ids=admins,
        timezone=timezone,
        db_path=db_path,
        default_fine=default_fine,
    )


class DB:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                full_name TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                event_at TEXT NOT NULL,
                end_at TEXT,
                action_text TEXT NOT NULL,
                week_start TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                source TEXT NOT NULL DEFAULT 'manual'
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL UNIQUE,
                reviewer_id INTEGER NOT NULL,
                compliant INTEGER NOT NULL,
                reviewed_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                event_id INTEGER NOT NULL,
                reviewer_id INTEGER NOT NULL,
                action_text TEXT NOT NULL,
                fine_amount INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                paid INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cur.execute("PRAGMA table_info(events)")
        event_columns = {row[1] for row in cur.fetchall()}
        if "source" not in event_columns:
            cur.execute("ALTER TABLE events ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'")
        if "end_at" not in event_columns:
            cur.execute("ALTER TABLE events ADD COLUMN end_at TEXT")
        self.conn.commit()

    def set_default_fine_if_missing(self, value: int) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES ('fine_amount', ?)",
            (str(value),),
        )
        self.conn.commit()

    def set_fine(self, amount: int) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO settings (key, value) VALUES ('fine_amount', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(amount),),
        )
        self.conn.commit()

    def get_fine(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM settings WHERE key='fine_amount'")
        row = cur.fetchone()
        return int(row["value"]) if row else 100

    def upsert_user(self, user_id: int, username: Optional[str], full_name: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO users (user_id, username, full_name) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET username=excluded.username, full_name=excluded.full_name",
            (user_id, username or "", full_name),
        )
        self.conn.commit()

    def clear_week_events(self, user_id: int, week_start: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM events WHERE user_id=? AND week_start=? AND status='pending' AND source='manual'",
            (user_id, week_start),
        )
        self.conn.commit()

    def insert_events(
        self,
        user_id: int,
        events: Iterable[tuple[datetime, Optional[datetime], str]],
        source: str = "manual",
    ) -> list[int]:
        cur = self.conn.cursor()
        ids: list[int] = []
        for event_at, end_at, action in events:
            week_start = (event_at.date() - timedelta(days=event_at.weekday())).isoformat()
            cur.execute(
                "INSERT INTO events (user_id, event_at, end_at, action_text, week_start, source) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, event_at.isoformat(), end_at.isoformat() if end_at else None, action, week_start, source),
            )
            ids.append(cur.lastrowid)
        self.conn.commit()
        return ids

    def get_event(self, event_id: int) -> Optional[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM events WHERE id=?", (event_id,))
        return cur.fetchone()

    def get_pending_future_events(self, now: datetime) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM events WHERE status='pending' AND event_at >= ?",
            (now.isoformat(),),
        )
        return cur.fetchall()

    def has_active_calendar(self, user_id: int, now: datetime) -> bool:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT 1 FROM events WHERE user_id=? AND status='pending' AND event_at >= ? LIMIT 1",
            (user_id, now.isoformat()),
        )
        return cur.fetchone() is not None

    def get_day_events(self, user_id: int, day_start: datetime, day_end: datetime) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM events
            WHERE user_id=? AND event_at >= ? AND event_at < ?
            ORDER BY event_at
            """,
            (user_id, day_start.isoformat(), day_end.isoformat()),
        )
        return cur.fetchall()

    def mark_review(self, event_id: int, reviewer_id: int, compliant: bool) -> bool:
        event = self.get_event(event_id)
        if not event or event["status"] != "pending":
            return False

        status = "ok" if compliant else "violation"
        now = datetime.utcnow().isoformat()
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE events SET status=? WHERE id=?",
            (status, event_id),
        )
        cur.execute(
            "INSERT INTO reviews (event_id, reviewer_id, compliant, reviewed_at) VALUES (?, ?, ?, ?)",
            (event_id, reviewer_id, int(compliant), now),
        )
        self.conn.commit()
        return True

    def create_violation(
        self, user_id: int, event_id: int, reviewer_id: int, action_text: str, fine_amount: int
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO violations (user_id, event_id, reviewer_id, action_text, fine_amount, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, event_id, reviewer_id, action_text, fine_amount, datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def get_user_stats(self, user_id: int) -> dict[str, int]:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM events WHERE user_id=?", (user_id,))
        total_events = int(cur.fetchone()["cnt"])

        cur.execute(
            "SELECT COUNT(*) AS cnt, COALESCE(SUM(fine_amount), 0) AS sum_fines "
            "FROM violations WHERE user_id=?",
            (user_id,),
        )
        row = cur.fetchone()
        violations = int(row["cnt"])
        fines = int(row["sum_fines"])

        cur.execute(
            "SELECT COUNT(*) AS cnt FROM violations WHERE user_id=? AND paid=0",
            (user_id,),
        )
        unpaid = int(cur.fetchone()["cnt"])

        return {
            "events": total_events,
            "violations": violations,
            "fines": fines,
            "unpaid_violations": unpaid,
        }

    def get_violations_breakdown(self, user_id: int, limit: int = 5) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT action_text, COUNT(*) AS cnt
            FROM violations
            WHERE user_id=?
            GROUP BY action_text
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return cur.fetchall()

    def get_bank(self) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT u.user_id, u.full_name,
                   COALESCE(SUM(v.fine_amount), 0) AS total_fines,
                   COALESCE(SUM(CASE WHEN v.paid=0 THEN v.fine_amount END), 0) AS unpaid_fines
            FROM users u
            LEFT JOIN violations v ON v.user_id = u.user_id
            GROUP BY u.user_id, u.full_name
            ORDER BY u.user_id
            """
        )
        return cur.fetchall()

    def get_user_name(self, user_id: int) -> str:
        cur = self.conn.cursor()
        cur.execute("SELECT full_name FROM users WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        return row["full_name"] if row else str(user_id)

    def mark_all_old_unreviewed_violation(self, older_than: datetime) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM events WHERE status='pending' AND event_at < ?",
            (older_than.isoformat(),),
        )
        rows = cur.fetchall()
        if not rows:
            return []
        cur.execute(
            "UPDATE events SET status='missed_review' WHERE status='pending' AND event_at < ?",
            (older_than.isoformat(),),
        )
        self.conn.commit()
        return rows


def allowed_only(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        cfg: Config = context.bot_data["config"]
        uid = update.effective_user.id if update.effective_user else None
        if uid not in cfg.allowed_user_ids:
            if update.effective_message:
                await update.effective_message.reply_text("Доступ закрыт.")
            return ConversationHandler.END
        return await func(update, context)

    return wrapper


def admin_only(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        cfg: Config = context.bot_data["config"]
        uid = update.effective_user.id if update.effective_user else None
        if uid not in cfg.admin_user_ids:
            if update.effective_message:
                await update.effective_message.reply_text("Только для админа.")
            return ConversationHandler.END
        return await func(update, context)

    return wrapper


def get_partner_id(cfg: Config, user_id: int) -> int:
    a, b = cfg.allowed_user_ids
    return b if user_id == a else a


def parse_time_range(left: str, now_year: int, tz: ZoneInfo) -> tuple[datetime, Optional[datetime]]:
    left = left.strip().replace("–", "-").replace("—", "-")
    for fmt in ("%Y-%m-%d %H:%M", "%d.%m %H:%M"):
        try:
            dt = datetime.strptime(left, fmt)
            if fmt == "%d.%m %H:%M":
                dt = dt.replace(year=now_year)
            return dt.replace(tzinfo=tz), None
        except ValueError:
            continue

    for date_fmt in ("%Y-%m-%d", "%d.%m"):
        try:
            date_part, time_range = left.split(" ", 1)
            start_raw, end_raw = [p.strip() for p in time_range.split("-", 1)]
            date_obj = datetime.strptime(date_part, date_fmt)
            if date_fmt == "%d.%m":
                date_obj = date_obj.replace(year=now_year)
            start_t = datetime.strptime(start_raw, "%H:%M").time()
            end_t = datetime.strptime(end_raw, "%H:%M").time()
            start_dt = datetime.combine(date_obj.date(), start_t, tzinfo=tz)
            end_dt = datetime.combine(date_obj.date(), end_t, tzinfo=tz)
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)
            return start_dt, end_dt
        except ValueError:
            continue

    raise ValueError(
        "дата/время должны быть в формате YYYY-MM-DD HH:MM, DD.MM HH:MM "
        "или с диапазоном YYYY-MM-DD HH:MM-HH:MM"
    )


def parse_calendar_lines(text: str, tz: ZoneInfo) -> list[tuple[datetime, Optional[datetime], str]]:
    events: list[tuple[datetime, Optional[datetime], str]] = []
    now_year = datetime.now(tz).year
    for idx, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "|" not in line:
            raise ValueError(f"Строка {idx}: нужен разделитель |")
        left, action = line.split("|", 1)
        left = left.strip()
        action = action.strip()
        if not action:
            raise ValueError(f"Строка {idx}: пустое действие")

        try:
            start_dt, end_dt = parse_time_range(left, now_year, tz)
        except ValueError as exc:
            raise ValueError(f"Строка {idx}: {exc}") from exc
        events.append((start_dt, end_dt, action))
    if not events:
        raise ValueError("Календарь пустой")
    return events


def unfold_ics_lines(content: str) -> list[str]:
    unfolded: list[str] = []
    for line in content.splitlines():
        if (line.startswith(" ") or line.startswith("\t")) and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return unfolded


def parse_ics_dt(raw: str, tzid: Optional[str], default_tz: ZoneInfo) -> datetime:
    value = raw.strip()
    if "T" in value:
        if value.endswith("Z"):
            dt = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=ZoneInfo("UTC"))
            return dt.astimezone(default_tz)
        dt = datetime.strptime(value, "%Y%m%dT%H%M%S")
        tz = ZoneInfo(tzid) if tzid else default_tz
        return dt.replace(tzinfo=tz).astimezone(default_tz)
    dt = datetime.strptime(value, "%Y%m%d")
    tz = ZoneInfo(tzid) if tzid else default_tz
    return dt.replace(tzinfo=tz)


def parse_ics_duration(raw: str) -> Optional[timedelta]:
    value = raw.strip()
    if not value.startswith("P"):
        return None
    if "T" not in value:
        return None
    date_part, time_part = value[1:].split("T", 1)
    if date_part:
        return None

    hours = 0
    minutes = 0
    seconds = 0
    token = ""
    for ch in time_part:
        if ch.isdigit():
            token += ch
            continue
        if not token:
            return None
        n = int(token)
        token = ""
        if ch == "H":
            hours = n
        elif ch == "M":
            minutes = n
        elif ch == "S":
            seconds = n
        else:
            return None
    if token:
        return None
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def parse_ics_calendar(ics_bytes: bytes, tz: ZoneInfo) -> list[tuple[datetime, Optional[datetime], str]]:
    try:
        content = ics_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = ics_bytes.decode("utf-8-sig", errors="replace")

    lines = unfold_ics_lines(content)
    events: list[tuple[datetime, Optional[datetime], str]] = []
    in_event = False
    dtstart_raw: Optional[str] = None
    dtstart_tzid: Optional[str] = None
    dtend_raw: Optional[str] = None
    dtend_tzid: Optional[str] = None
    duration_raw: Optional[str] = None
    rrule_raw: Optional[str] = None
    summary = ""
    now = datetime.now(tz)
    window_start = datetime.combine(now.date(), time.min, tzinfo=tz)
    window_end = window_start + timedelta(days=7)

    for line in lines:
        if line == "BEGIN:VEVENT":
            in_event = True
            dtstart_raw = None
            dtstart_tzid = None
            dtend_raw = None
            dtend_tzid = None
            duration_raw = None
            rrule_raw = None
            summary = ""
            continue
        if line == "END:VEVENT":
            if dtstart_raw:
                event_dt = parse_ics_dt(dtstart_raw, dtstart_tzid, tz)
                end_dt = parse_ics_dt(dtend_raw, dtend_tzid, tz) if dtend_raw else None
                if end_dt is None and duration_raw:
                    duration = parse_ics_duration(duration_raw)
                    if duration:
                        end_dt = event_dt + duration
                action = summary.strip() or "Событие"
                if rrule_raw:
                    duration_delta = (end_dt - event_dt) if end_dt else None
                    try:
                        rule = rrulestr(rrule_raw, dtstart=event_dt)
                        occ_starts = rule.between(window_start, window_end, inc=True)
                        for occ_start in occ_starts:
                            if occ_start.tzinfo is None:
                                occ_start = occ_start.replace(tzinfo=event_dt.tzinfo)
                            occ_end = occ_start + duration_delta if duration_delta else None
                            events.append((occ_start, occ_end, action))
                    except Exception:
                        events.append((event_dt, end_dt, action))
                else:
                    events.append((event_dt, end_dt, action))
            in_event = False
            continue
        if not in_event:
            continue

        if line.startswith("DTSTART"):
            head, _, value = line.partition(":")
            dtstart_raw = value.strip()
            if ";" in head:
                params = head.split(";")[1:]
                for p in params:
                    if p.startswith("TZID="):
                        dtstart_tzid = p.split("=", 1)[1]
                        break
        elif line.startswith("DTEND"):
            head, _, value = line.partition(":")
            dtend_raw = value.strip()
            if ";" in head:
                params = head.split(";")[1:]
                for p in params:
                    if p.startswith("TZID="):
                        dtend_tzid = p.split("=", 1)[1]
                        break
        elif line.startswith("DURATION:"):
            duration_raw = line.split(":", 1)[1].strip()
        elif line.startswith("RRULE:"):
            rrule_raw = line.split(":", 1)[1].strip()
        elif line.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()

    if not events:
        raise ValueError("В .ics не найдено событий VEVENT с DTSTART")
    return events


def format_day_tasks(
    db: DB, user_id: int, target_date: datetime, tz: ZoneInfo, title: str
) -> str:
    day_start = datetime.combine(target_date.date(), time.min, tzinfo=tz)
    day_end = day_start + timedelta(days=1)
    rows = db.get_day_events(user_id, day_start, day_end)
    if not rows:
        return f"{title}: дел на сегодня нет."

    lines = [f"{title} ({day_start.strftime('%Y-%m-%d')}):"]
    for row in rows:
        lines.append(f"- {format_event_window(row['event_at'], row['end_at'], tz)} {row['action_text']}")
    return "\n".join(lines)


def split_events_by_week(
    events: list[tuple[datetime, Optional[datetime], str]]
) -> dict[str, list[tuple[datetime, Optional[datetime], str]]]:
    grouped: dict[str, list[tuple[datetime, Optional[datetime], str]]] = {}
    for event_dt, end_dt, action in events:
        week_start = (event_dt.date() - timedelta(days=event_dt.weekday())).isoformat()
        grouped.setdefault(week_start, []).append((event_dt, end_dt, action))
    return grouped


def filter_ics_to_actual_week(
    events: list[tuple[datetime, Optional[datetime], str]], now: datetime, tz: ZoneInfo
) -> list[tuple[datetime, Optional[datetime], str]]:
    start = datetime.combine(now.date(), time.min, tzinfo=tz)
    end = start + timedelta(days=7)
    return [
        (event_dt, end_dt, action)
        for event_dt, end_dt, action in events
        if start <= event_dt < end
    ]


def format_event_window(event_at_iso: str, end_at_iso: Optional[str], tz: ZoneInfo) -> str:
    start_dt = datetime.fromisoformat(event_at_iso).astimezone(tz)
    if not end_at_iso:
        return start_dt.strftime("%H:%M")
    end_dt = datetime.fromisoformat(end_at_iso).astimezone(tz)
    return f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"


def format_stats(db: DB, user_id: int) -> str:
    stats = db.get_user_stats(user_id)
    top = db.get_violations_breakdown(user_id)
    lines = [
        f"Событий: {stats['events']}",
        f"Нарушений: {stats['violations']}",
        f"Штрафов, руб: {stats['fines']}",
        f"Незакрытых нарушений: {stats['unpaid_violations']}",
    ]
    if top:
        lines.append("Топ нарушаемых действий:")
        for row in top:
            lines.append(f"- {row['action_text']}: {row['cnt']}")
    return "\n".join(lines)


def panel_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Установить штраф", callback_data="panel:set_fine")],
            [InlineKeyboardButton("Показать банк", callback_data="panel:bank")],
            [InlineKeyboardButton("Моя статистика", callback_data="panel:my_stats")],
            [InlineKeyboardButton("Статистика партнера", callback_data="panel:other_stats")],
        ]
    )


@allowed_only
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = context.bot_data["config"]
    db: DB = context.bot_data["db"]
    user = update.effective_user
    db.upsert_user(user.id, user.username, user.full_name)
    now = datetime.now(cfg.timezone)
    partner_id = get_partner_id(cfg, user.id)
    my_active = db.has_active_calendar(user.id, now)
    other_active = db.has_active_calendar(partner_id, now)

    text = (
        "Команды:\n"
        "/upload - загрузить календарь на неделю\n"
        "/today - список дел на сегодня (свои и коллеги)\n"
        "/panel - админ-панель\n"
        "/stats - моя статистика\n"
        "/stats_other - статистика партнера\n"
        "/bank - банк по штрафам\n"
        "/myid - показать мой Telegram ID\n"
        "/help - справка по формату загрузки"
    )
    status_lines = ["", "Статус календарей:"]
    status_lines.append("Твой календарь: активен" if my_active else "Твой календарь: нет активного")
    status_lines.append(
        "Календарь коллеги: активен" if other_active else "Календарь коллеги: нет активного"
    )
    if not my_active and not other_active:
        status_lines.append("Ни у кого нет активного календаря. Загрузите новый через /upload.")

    today_lines = [
        "",
        format_day_tasks(db, user.id, now, cfg.timezone, "Твои дела"),
        format_day_tasks(
            db,
            partner_id,
            now,
            cfg.timezone,
            f"Дела коллеги ({db.get_user_name(partner_id)})",
        ),
    ]

    await update.message.reply_text(text + "\n".join(status_lines + today_lines))


@allowed_only
async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Твой Telegram ID: {update.effective_user.id}")


@allowed_only
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Загрузка календаря:\n"
        "1) Текстом, каждая строка отдельно:\n"
        "YYYY-MM-DD HH:MM | Действие\n"
        "или YYYY-MM-DD HH:MM-HH:MM | Действие\n"
        "или\n"
        "DD.MM HH:MM | Действие\n"
        "или DD.MM HH:MM-HH:MM | Действие\n"
        "2) Файлом .ics (экспорт из календаря)\n\n"
        "Пример:\n"
        "2026-02-16 08:00 | Зарядка\n"
        "16.02 09:00-10:30 | Тренировка\n\n"
        "Если отправишь несколько недель сразу, бот сам разложит их по неделям.\n"
        "Если расписание не изменилось, можно не загружать заново."
    )


@allowed_only
async def upload_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Отправь календарь:\n"
        "- текстом (YYYY-MM-DD HH:MM | Действие или HH:MM-HH:MM)\n"
        "- или файлом .ics\n\n"
        "Можно отправить сразу несколько недель, бот сам разобьет и сохранит.\n"
        "Если календарь остался прежним, можно ничего не отправлять.\n"
        "Текущий календарь сохранится."
    )
    return UPLOAD_CALENDAR


async def save_uploaded_events(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    events: list[tuple[datetime, Optional[datetime], str]],
) -> int:
    cfg: Config = context.bot_data["config"]
    db: DB = context.bot_data["db"]
    user_id = update.effective_user.id
    grouped = split_events_by_week(events)

    week_count = 0
    for week_start, week_events in sorted(grouped.items()):
        db.clear_week_events(user_id, week_start)
        event_ids = db.insert_events(user_id, week_events)
        for event_id, (event_dt, _end_dt, action) in zip(event_ids, week_events):
            schedule_event_notification(context.application, cfg, event_id, event_dt, action)
        week_count += 1

    await update.message.reply_text(
        f"Сохранено событий: {len(events)}\nОбновлено недель: {week_count}"
    )
    return ConversationHandler.END


@allowed_only
async def upload_receive(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    cfg: Config = context.bot_data["config"]
    if update.message.document:
        doc = update.message.document
        name = (doc.file_name or "").lower()
        if not name.endswith(".ics"):
            await update.message.reply_text("Поддерживается только .ics файл.")
            return UPLOAD_CALENDAR
        file = await context.bot.get_file(doc.file_id)
        payload = await file.download_as_bytearray()
        try:
            events = parse_ics_calendar(bytes(payload), cfg.timezone)
            now = datetime.now(cfg.timezone)
            events = filter_ics_to_actual_week(events, now, cfg.timezone)
            if not events:
                await update.message.reply_text(
                    "В .ics нет актуальных событий на период от текущего дня до +7 дней."
                )
                return UPLOAD_CALENDAR
        except ValueError as exc:
            await update.message.reply_text(f"Ошибка .ics: {exc}")
            return UPLOAD_CALENDAR
        return await save_uploaded_events(update, context, events)

    text = update.message.text or ""
    try:
        events = parse_calendar_lines(text, cfg.timezone)
    except ValueError as exc:
        await update.message.reply_text(f"Ошибка: {exc}")
        return UPLOAD_CALENDAR
    return await save_uploaded_events(update, context, events)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Отменено.")
    return ConversationHandler.END


@allowed_only
@admin_only
async def panel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Админ-панель", reply_markup=panel_keyboard())


@allowed_only
@admin_only
async def set_fine_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["awaiting_fine"] = True
    await update.message.reply_text("Отправь новую стоимость штрафа в рублях (целое число).")


@allowed_only
@admin_only
async def set_fine_receive(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: DB = context.bot_data["db"]
    text = (update.message.text or "").strip()
    if not text.isdigit():
        await update.message.reply_text("Нужно положительное целое число.")
        return
    amount = int(text)
    if amount <= 0:
        await update.message.reply_text("Сумма должна быть больше 0.")
        return
    db.set_fine(amount)
    context.user_data["awaiting_fine"] = False
    await update.message.reply_text(f"Новый штраф: {amount} руб.")


@allowed_only
@admin_only
async def maybe_receive_fine(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("awaiting_fine"):
        return
    await set_fine_receive(update, context)


@allowed_only
async def stats_me(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: DB = context.bot_data["db"]
    user_id = update.effective_user.id
    await update.message.reply_text("Твоя статистика:\n" + format_stats(db, user_id))


@allowed_only
async def stats_other(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = context.bot_data["config"]
    db: DB = context.bot_data["db"]
    user_id = update.effective_user.id
    partner_id = get_partner_id(cfg, user_id)
    name = db.get_user_name(partner_id)
    await update.message.reply_text(f"Статистика {name}:\n" + format_stats(db, partner_id))


@allowed_only
async def today_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = context.bot_data["config"]
    db: DB = context.bot_data["db"]
    user_id = update.effective_user.id
    partner_id = get_partner_id(cfg, user_id)
    now = datetime.now(cfg.timezone)

    text = "\n\n".join(
        [
            format_day_tasks(db, user_id, now, cfg.timezone, "Твои дела"),
            format_day_tasks(
                db,
                partner_id,
                now,
                cfg.timezone,
                f"Дела коллеги ({db.get_user_name(partner_id)})",
            ),
        ]
    )
    await update.message.reply_text(text)


@allowed_only
@admin_only
async def bank(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db: DB = context.bot_data["db"]
    rows = db.get_bank()
    if not rows:
        await update.message.reply_text("Банк пока пуст.")
        return
    lines = ["Банк по штрафам:"]
    for row in rows:
        lines.append(
            f"{row['full_name']} ({row['user_id']}): всего {row['total_fines']} руб, долг {row['unpaid_fines']} руб"
        )
    await update.message.reply_text("\n".join(lines))


@allowed_only
async def panel_callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    cfg: Config = context.bot_data["config"]
    db: DB = context.bot_data["db"]
    user_id = query.from_user.id

    if user_id not in cfg.admin_user_ids:
        await query.message.reply_text("Только для админа.")
        return

    data = query.data
    if data == "panel:set_fine":
        context.user_data["awaiting_fine"] = True
        await query.message.reply_text("Отправь новую сумму штрафа числом.")
        return
    if data == "panel:bank":
        rows = db.get_bank()
        if not rows:
            await query.message.reply_text("Банк пока пуст.")
            return
        lines = ["Банк по штрафам:"]
        for row in rows:
            lines.append(
                f"{row['full_name']} ({row['user_id']}): всего {row['total_fines']} руб, долг {row['unpaid_fines']} руб"
            )
        await query.message.reply_text("\n".join(lines))
        return
    if data == "panel:my_stats":
        await query.message.reply_text("Твоя статистика:\n" + format_stats(db, user_id))
        return
    if data == "panel:other_stats":
        partner_id = get_partner_id(cfg, user_id)
        name = db.get_user_name(partner_id)
        await query.message.reply_text(f"Статистика {name}:\n" + format_stats(db, partner_id))
        return


def schedule_event_notification(
    application: Application, cfg: Config, event_id: int, event_dt: datetime, action: str
) -> None:
    job_name = f"event:{event_id}"
    for job in application.job_queue.get_jobs_by_name(job_name):
        job.schedule_removal()

    if event_dt <= datetime.now(cfg.timezone):
        return
    application.job_queue.run_once(
        send_event_notification_job,
        when=event_dt,
        name=job_name,
        data={"event_id": event_id, "action": action},
    )


async def send_event_notification_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = context.application.bot_data["config"]
    db: DB = context.application.bot_data["db"]
    event_id = context.job.data["event_id"]
    event = db.get_event(event_id)
    if not event or event["status"] != "pending":
        return

    owner_id = int(event["user_id"])
    partner_id = get_partner_id(cfg, owner_id)
    action = event["action_text"]
    window = format_event_window(event["event_at"], event["end_at"], cfg.timezone)
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Да", callback_data=f"review:{event_id}:1"),
                InlineKeyboardButton("Нет", callback_data=f"review:{event_id}:0"),
            ]
        ]
    )
    text = (
        f"Проверка события {db.get_user_name(owner_id)}:\n"
        f"{window} - {action}\n"
        "Соблюдает график?"
    )
    await context.bot.send_message(chat_id=partner_id, text=text, reply_markup=keyboard)


@allowed_only
async def review_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    cfg: Config = context.bot_data["config"]
    db: DB = context.bot_data["db"]

    try:
        _, event_id_str, compliant_str = query.data.split(":")
        event_id = int(event_id_str)
        compliant = bool(int(compliant_str))
    except (ValueError, AttributeError):
        await query.message.reply_text("Некорректный callback.")
        return

    event = db.get_event(event_id)
    if not event:
        await query.message.reply_text("Событие не найдено.")
        return

    owner_id = int(event["user_id"])
    expected_reviewer = get_partner_id(cfg, owner_id)
    if query.from_user.id != expected_reviewer:
        await query.message.reply_text("Только второй игрок может подтвердить.")
        return

    ok = db.mark_review(event_id, query.from_user.id, compliant)
    if not ok:
        await query.message.reply_text("Это событие уже оценено.")
        return

    action = event["action_text"]
    window = format_event_window(event["event_at"], event["end_at"], cfg.timezone)
    if compliant:
        await context.bot.send_message(
            chat_id=owner_id,
            text=(
                f"Событие отмечено как выполненное: "
                f"{window} - {action}"
            ),
        )
        await query.message.reply_text("Отметка сохранена: соблюдает график.")
        return

    fine = db.get_fine()
    db.create_violation(owner_id, event_id, query.from_user.id, action, fine)
    await context.bot.send_message(
        chat_id=owner_id,
        text=(
            f"Нарушение: {window} - {action}\n"
            f"Штраф: {fine} руб.\n"
            "Нужно пополнить счёт и закрыть долг."
        ),
    )
    await query.message.reply_text("Нарушение зафиксировано.")


async def sunday_reminder_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = context.application.bot_data["config"]
    for uid in cfg.allowed_user_ids:
        await context.bot.send_message(
            chat_id=uid,
            text=(
                "Напоминание: сегодня воскресенье 19:00, проверь календарь на следующую неделю.\n"
                "Если расписание изменилось, загрузи новый через /upload.\n"
                "Если без изменений, можно ничего не загружать."
            ),
        )


async def mark_expired_events_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = context.application.bot_data["config"]
    db: DB = context.application.bot_data["db"]
    now = datetime.now(cfg.timezone)
    expired = db.mark_all_old_unreviewed_violation(now - timedelta(hours=12))
    if not expired:
        return

    fine = db.get_fine()
    for event in expired:
        owner_id = int(event["user_id"])
        reviewer_id = get_partner_id(cfg, owner_id)
        action = event["action_text"]
        db.create_violation(owner_id, int(event["id"]), reviewer_id, action, fine)
        window = format_event_window(event["event_at"], event["end_at"], cfg.timezone)
        await context.bot.send_message(
            chat_id=owner_id,
            text=(
                f"Событие не было подтверждено вовремя и засчитано как нарушение:\n"
                f"{window} - {action}\n"
                f"Штраф: {fine} руб."
            ),
        )


def restore_jobs(application: Application, cfg: Config, db: DB) -> None:
    now = datetime.now(cfg.timezone)
    for row in db.get_pending_future_events(now):
        event_dt = datetime.fromisoformat(row["event_at"]).astimezone(cfg.timezone)
        schedule_event_notification(
            application,
            cfg,
            int(row["id"]),
            event_dt,
            row["action_text"],
        )


async def post_init(application: Application) -> None:
    cfg: Config = application.bot_data["config"]
    db: DB = application.bot_data["db"]
    restore_jobs(application, cfg, db)

    application.job_queue.run_daily(
        sunday_reminder_job,
        time=time(hour=19, minute=0, tzinfo=cfg.timezone),
        days=(6,),
        name="weekly_sunday_reminder",
    )
    application.job_queue.run_repeating(
        mark_expired_events_job,
        interval=3600,
        first=10,
        name="expired_events_checker",
    )
    logger.info("Bot initialized")


def main() -> None:
    load_dotenv()
    cfg = load_config()
    db = DB(cfg.db_path)
    db.set_default_fine_if_missing(cfg.default_fine)

    application = Application.builder().token(cfg.token).post_init(post_init).build()
    application.bot_data["config"] = cfg
    application.bot_data["db"] = db

    upload_conv = ConversationHandler(
        entry_points=[CommandHandler("upload", upload_start)],
        states={
            UPLOAD_CALENDAR: [
                MessageHandler((filters.TEXT | filters.Document.ALL) & ~filters.COMMAND, upload_receive)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("myid", myid))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(upload_conv)
    application.add_handler(CommandHandler("setfine", set_fine_start))
    application.add_handler(CommandHandler("panel", panel))
    application.add_handler(CommandHandler("stats", stats_me))
    application.add_handler(CommandHandler("stats_other", stats_other))
    application.add_handler(CommandHandler("today", today_tasks))
    application.add_handler(CommandHandler("bank", bank))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, maybe_receive_fine))
    application.add_handler(CallbackQueryHandler(panel_callbacks, pattern=r"^panel:"))
    application.add_handler(CallbackQueryHandler(review_callback, pattern=r"^review:"))

    application.run_polling()


if __name__ == "__main__":
    main()

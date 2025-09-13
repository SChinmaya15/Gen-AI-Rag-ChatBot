import sqlite3
from typing import List, Tuple, Optional, Any

DB_NAME = "bot_data.db"

def _conn():
    # check_same_thread=False lets us use the DB from Flask + bot in dev
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = _conn()
    c = conn.cursor()

    # Conversations log
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        message TEXT,
        bot_reply TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Tickets created
    c.execute("""
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        software TEXT,
        ticket_id TEXT,
        execution_id TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def save_conversation(user_id: str, message: str, bot_reply: str) -> None:
    conn = _conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (user_id, message, bot_reply) VALUES (?, ?, ?)",
        (user_id, message, bot_reply),
    )
    conn.commit()
    conn.close()

def save_ticket(user_id: str, software: str, ticket_id: str, execution_id: str) -> None:
    conn = _conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO tickets (user_id, software, ticket_id, execution_id) VALUES (?, ?, ?, ?)",
        (user_id, software, ticket_id, execution_id),
    )
    conn.commit()
    conn.close()

def get_tickets(user_id: str) -> List[Tuple[Any, ...]]:
    conn = _conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, user_id, software, ticket_id, execution_id, created_at "
        "FROM tickets WHERE user_id=? ORDER BY created_at DESC",
        (user_id,),
    )
    rows = c.fetchall()
    conn.close()
    return rows

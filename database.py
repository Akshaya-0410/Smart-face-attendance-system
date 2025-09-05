import sqlite3
import time

# Initialize the database with necessary tables
def initialize_database():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    ''')

    # Create attendance table with slot support
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            slot TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()

# Register a new user (with overwrite support in logic)
def register_user(user_id, name):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Check if user exists
    cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
    existing_user = cursor.fetchone()

    if existing_user:
        conn.close()
        return "User ID already exists!"

    cursor.execute("INSERT INTO users (id, name) VALUES (?, ?)", (user_id, name))
    conn.commit()
    conn.close()
    return "User registered successfully!"

# Record attendance (with duplicate prevention per slot)
def record_attendance(user_id, user_name, slot):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    date = time.strftime("%d-%B-%Y")
    time_now = time.strftime("%H:%M:%S")

    cursor.execute("SELECT * FROM attendance WHERE user_id=? AND date=? AND slot=?", (user_id, date, slot))
    if cursor.fetchone():
        conn.close()
        return date, time_now, False  # Already marked

    cursor.execute("INSERT INTO attendance (user_id, name, date, time, slot) VALUES (?, ?, ?, ?, ?)",
                   (user_id, user_name, date, time_now, slot))
    conn.commit()
    conn.close()
    return date, time_now, True  # Newly marked

# Get all registered users
def get_all_users():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    conn.close()
    return users

# Get all attendance records
def get_attendance_records():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC")
    records = cursor.fetchall()
    conn.close()
    return records

# Ensure tables are created when the module is imported
initialize_database()

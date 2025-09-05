import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sqlite3
import os
import time

# Ensure database exists before querying
def initialize_database():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

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


# Call initialization before querying
initialize_database()
# Initialize main window
window = tk.Tk()
window.title("Smart Face Attendance System")
window.geometry("1280x720")
window.configure(bg="#C6DBEF")  # Light blue background

# Title Header
header = tk.Label(
    window,
    text="Smart Face Attendance System",
    bg="#1C1C1C", fg="white",
    font=("Poppins", 24, "bold"),
    height=2
)
header.pack(fill="both")

# Time Display
def update_time():
    current_time = time.strftime("%d-%B-%Y | %H:%M:%S")
    time_label.config(text=current_time)
    window.after(1000, update_time)

time_label = tk.Label(
    window, text="", bg="#6BAED6", fg="black",
    font=("Poppins", 16, "bold")
)
time_label.pack(fill="both")
update_time()

# Frames
left_frame = tk.Frame(window, bg="#9ECAE1", relief="ridge", borderwidth=3)
left_frame.place(relx=0.05, rely=0.2, relwidth=0.42, relheight=0.7)

right_frame = tk.Frame(window, bg="#9ECAE1", relief="ridge", borderwidth=3)
right_frame.place(relx=0.53, rely=0.2, relwidth=0.42, relheight=0.7)

# Left Section (For Attendance Tracking)
title1 = tk.Label(
    left_frame, text="Attendance Tracking", bg="#DEEBF7",
    fg="black", font=("Poppins", 18, "bold")
)
title1.pack(fill="both", pady=10)

# Table for Attendance Data
table_frame = tk.Frame(left_frame, bg="white", relief="ridge", borderwidth=2)
table_frame.pack(pady=10, padx=10, fill="both", expand=True)

columns = ("ID", "NAME", "DATE", "TIME", "SLOT")

tree = ttk.Treeview(table_frame, columns=columns, show="headings")
for col in columns:
    tree.heading(col, text=col, anchor="center")
    tree.column(col, width=80, anchor="center")
tree.pack(fill="both", expand=True)

# Function to Load Attendance Data
def load_attendance():
    tree.delete(*tree.get_children())  # Clear existing data
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    # Ensure tables exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            user_id INTEGER,
            name TEXT,
            date TEXT,
            time TEXT,
            slot TEXT
        )
    ''')
    conn.commit()
    today = time.strftime("%d-%B-%Y")
    cursor.execute("SELECT user_id, name, date, time, slot FROM attendance WHERE date=? ORDER BY time DESC", (today,))


    rows = cursor.fetchall()
    for row in rows:
        tree.insert("", "end", values=row)
    conn.close()

# Button to Refresh Attendance Data
refresh_btn = tk.Button(left_frame, text="Refresh Attendance", font=("Poppins", 14, "bold"), bg="#1C1C1C",
                        fg="white", relief="ridge", borderwidth=3, command=load_attendance)
refresh_btn.pack(pady=10)

# Button to Open `track.py` for Attendance Tracking
def open_track():
    subprocess.Popen(["python", "track.py"])

take_attendance_btn = tk.Button(
    left_frame, text="Open Attendance Tracking",
    font=("Poppins", 16, "bold"), bg="#1C1C1C", fg="white",
    width=25, height=2, relief="ridge", borderwidth=3,
    command=open_track
)
take_attendance_btn.pack(pady=15)

# Right Section (For New Registrations)
title2 = tk.Label(
    right_frame, text="New Registrations", bg="#DEEBF7",
    fg="black", font=("Poppins", 18, "bold")
)
title2.pack(fill="both", pady=10)

# Table for Registered Users
reg_table_frame = tk.Frame(right_frame, bg="white", relief="ridge", borderwidth=2)
reg_table_frame.pack(pady=10, padx=10, fill="both", expand=True)

reg_columns = ("ID", "NAME")
reg_tree = ttk.Treeview(reg_table_frame, columns=reg_columns, show="headings")
for col in reg_columns:
    reg_tree.heading(col, text=col, anchor="center")
    reg_tree.column(col, width=100, anchor="center")
reg_tree.pack(fill="both", expand=True)

# Function to Load Registered Users
def load_registered_users():
    reg_tree.delete(*reg_tree.get_children())  # Clear existing data
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    # Ensure tables exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            user_id INTEGER,
            name TEXT,
            date TEXT,
            time TEXT,
            slot TEXT
        )
    ''')
    conn.commit()
    cursor.execute("SELECT id, name FROM users ORDER BY id")

    rows = cursor.fetchall()
    for row in rows:
        reg_tree.insert("", "end", values=row)
    conn.close()

# Button to Refresh Registered Users
refresh_users_btn = tk.Button(right_frame, text="Refresh Users", font=("Poppins", 14, "bold"), bg="#1C1C1C",fg="white", relief="ridge", borderwidth=3, command=load_registered_users)
refresh_users_btn.pack(pady=10)

# Button to Open `Track1.py` for New Registrations
def open_registration():
    subprocess.Popen(["python", "Track1.py"])

register_btn = tk.Button(
    right_frame, text="Open New Registration",
    font=("Poppins", 16, "bold"), bg="#1C1C1C", fg="white",
    width=25, height=2, relief="ridge", borderwidth=3,
    command=open_registration
)
register_btn.pack(pady=15)

# Quit Button
quit_btn = tk.Button(
    window, text="Quit",
    font=("Poppins", 12, "bold"), bg="#D32F2F", fg="white",  # Smaller font and different color
    width=15, height=1, relief="ridge", borderwidth=3,
    command=window.destroy
)
quit_btn.place(relx=0.5, rely=0.95, anchor="center")  # Position at the bottom center

def show_attendance_report():
    report_window = tk.Toplevel(window)
    report_window.title("Attendance Report")
    report_window.geometry("1000x650")
    report_window.configure(bg="#E5F5FD")

    title = tk.Label(report_window, text="Attendance Report", bg="#2171B5", fg="white",
                     font=("Poppins", 20, "bold"))
    title.pack(fill="x", pady=10)

    top_frame = tk.Frame(report_window, bg="#E5F5FD")
    top_frame.pack(fill="x", padx=20, pady=5)

    mode_var = tk.StringVar(value="Full")  # Default = Full Attendance

    # Full vs Slotwise radio buttons
    full_radio = tk.Radiobutton(top_frame, text="Full Attendance", variable=mode_var, value="Full",
                                font=("Poppins", 12), bg="#E5F5FD", command=lambda: update_table())
    slot_radio = tk.Radiobutton(top_frame, text="Slotwise Attendance", variable=mode_var, value="Slotwise",
                                font=("Poppins", 12), bg="#E5F5FD", command=lambda: update_table())
    full_radio.pack(side="left", padx=10)
    slot_radio.pack(side="left", padx=10)

    slot_var = tk.StringVar(value="Morning")
    slot_options = ["Morning", "Afternoon", "Evening"]
    slot_menu = ttk.Combobox(top_frame, textvariable=slot_var, values=slot_options, font=("Poppins", 12), width=12)
    slot_menu.pack(side="left", padx=10)

    show_btn = tk.Button(top_frame, text="Show Report", font=("Poppins", 12, "bold"),
                         bg="#1C1C1C", fg="white", command=lambda: update_table())
    show_btn.pack(side="left", padx=20)

    # Table
    table_frame = tk.Frame(report_window, bg="#E5F5FD")
    table_frame.pack(fill="both", expand=True, padx=20, pady=10)

    columns = ("ID", "Name", "Total Classes", "Days Present", "Days Absent")
    report_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
    for col in columns:
        report_tree.heading(col, text=col)
        report_tree.column(col, anchor="center", width=150)
    report_tree.pack(fill="both", expand=True)

    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=report_tree.yview)
    report_tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    close_btn = tk.Button(report_window, text="Close", command=report_window.destroy,
                          font=("Poppins", 12, "bold"), bg="#D32F2F", fg="white")
    close_btn.pack(pady=10)

    # Function to Update Table
    def update_table():
        for row in report_tree.get_children():
            report_tree.delete(row)

        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        # Ensure tables exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                user_id INTEGER,
                name TEXT,
                date TEXT,
                time TEXT,
                slot TEXT
            )
        ''')
        conn.commit()
        if mode_var.get() == "Full":
            cursor.execute("SELECT DISTINCT date FROM attendance")
            total_dates = set([row[0] for row in cursor.fetchall()])
            total_classes = len(total_dates)

            cursor.execute("""
                SELECT user_id, name, date
                FROM attendance
                GROUP BY user_id, date
            """)
            user_date_records = cursor.fetchall()

            user_present_days = {}
            user_names = {}

            for user_id, name, date in user_date_records:
                if user_id not in user_present_days:
                    user_present_days[user_id] = set()
                    user_names[user_id] = name
                user_present_days[user_id].add(date)

            for user_id, present_dates in user_present_days.items():
                days_present = len(present_dates)
                days_absent = total_classes - days_present
                name = user_names[user_id]
                report_tree.insert("", "end", values=(user_id, name, total_classes, days_present, days_absent))

        elif mode_var.get() == "Slotwise":
            selected_slot = slot_var.get()

            cursor.execute("SELECT DISTINCT date FROM attendance WHERE slot=?", (selected_slot,))
            total_dates = set([row[0] for row in cursor.fetchall()])
            total_classes = len(total_dates)

            cursor.execute("""
                SELECT user_id, name, date
                FROM attendance
                WHERE slot=?
                GROUP BY user_id, date
            """, (selected_slot,))
            user_date_records = cursor.fetchall()

            user_present_days = {}
            user_names = {}

            for user_id, name, date in user_date_records:
                if user_id not in user_present_days:
                    user_present_days[user_id] = set()
                    user_names[user_id] = name
                user_present_days[user_id].add(date)

            for user_id, present_dates in user_present_days.items():
                days_present = len(present_dates)
                days_absent = total_classes - days_present
                name = user_names[user_id]
                report_tree.insert("", "end", values=(user_id, name, total_classes, days_present, days_absent))

        conn.close()

    update_table()



report_btn = tk.Button(right_frame, text="Attendance % Report", font=("Poppins", 14, "bold"),
                       bg="#1C1C1C", fg="white",width=20,height=2, relief="ridge", borderwidth=3,
                       command=show_attendance_report)
report_btn.pack(pady=3)

# Load data initially
load_attendance()
load_registered_users()

window.mainloop()

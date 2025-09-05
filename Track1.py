import cv2
import os
import sqlite3
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle
from deepface import DeepFace
# Create dataset folder if not exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Create or connect to database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Ensure users table exists
# Ensure attendance table exists
# Ensure required tables exist
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



# GUI setup
window = tk.Tk()
window.title("Smart Face Attendance - Registration")
window.geometry("1280x720")
window.configure(bg="#C6DBEF")

header = tk.Label(window, text="New User Registration", bg="#1C1C1C", fg="white",
                  font=("Poppins", 24, "bold"), height=2)
header.pack(fill="both")

reg_frame = tk.Frame(window, bg="#9ECAE1", relief="ridge", borderwidth=3)
reg_frame.place(relx=0.2, rely=0.2, relwidth=0.6, relheight=0.7)

title = tk.Label(reg_frame, text="Register New User", bg="#DEEBF7", fg="black",
                 font=("Poppins", 18, "bold"))
title.pack(fill="both", pady=10)

# ID Entry
id_label = tk.Label(reg_frame, text="Enter ID", font=("Poppins", 14), bg="#4292C6")
id_label.pack(pady=5)
id_entry = tk.Entry(reg_frame, font=("Poppins", 14), width=25, relief="ridge", borderwidth=3)
id_entry.pack()

# Name Entry
name_label = tk.Label(reg_frame, text="Enter Name", font=("Poppins", 14), bg="#4292C6")
name_label.pack(pady=5)
name_entry = tk.Entry(reg_frame, font=("Poppins", 14), width=25, relief="ridge", borderwidth=3)
name_entry.pack()



# Register logic
def capture_images():
    user_id = id_entry.get()
    user_name = name_entry.get()

    if not user_id or not user_name:
        messagebox.showwarning("Error", "Please enter both ID and Name.")
        return

    try:
        user_id = int(user_id)
    except:
        messagebox.showwarning("Error", "User ID must be a number.")
        return

    # Check if user already exists
    cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
    if cursor.fetchone():
        if not messagebox.askyesno("Overwrite", f"User ID {user_id} already exists.\nOverwrite images and details?"):
            return
        cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
        for file in os.listdir("dataset"):
            if file.startswith(f"{user_id}_"):
                os.remove(os.path.join("dataset", file))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam could not be opened.")
        return

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    messagebox.showinfo("Instructions", "Press 'C' to capture face images.\nCapture at least 3 clear faces.\nPress 'Q' to finish.")

    count = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            print("[ERROR] Empty frame captured. Skipping...")
            continue
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capture Face (Press C to Capture, Q to Quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(faces) == 0:  # Check if no faces are detected
                print("[WARNING] No face detected. Try again.")
            else:
                for (x, y, w, h) in faces:
                   face_crop = frame[y:y+h, x:x+w]
                   try:
                       face_crop = cv2.resize(face_crop, (128, 128))
                   except Exception as e:
                       print(f"[ERROR] Resize failed: {e}")
                       continue

                   count += 1
                   filename = f"dataset/{user_id}_{user_name}_{count}.jpg"
                   cv2.imwrite(filename, face_crop)
                   print(f"[Saved] {filename}")


        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count >= 1:
        cursor.execute("INSERT INTO users (id, name) VALUES (?, ?)", (user_id, user_name))
        conn.commit()
        messagebox.showinfo("Success", f"User {user_name} registered with {count} image(s).")
    else:
        messagebox.showwarning("Incomplete", "No valid face images were captured.")

# Button
register_btn = tk.Button(reg_frame, text="Take Images & Register", font=("Poppins", 16, "bold"),
                         bg="#1C1C1C", fg="white", width=20, height=2, relief="ridge", borderwidth=3,
                         command=capture_images)
register_btn.pack(pady=15)

# Exit button
exit_btn = tk.Button(reg_frame, text="Exit", font=("Poppins", 12, "bold"),
                     bg="#A94442", fg="white", width=10, command=window.destroy)
exit_btn.pack(pady=10)

window.mainloop()
conn.close()
import pickle
from deepface import DeepFace
import cv2
import os

def update_embeddings():
    print("[INFO] Updating face embeddings...")
    registered_faces = {}
    for file in os.listdir("dataset"):
        if file.endswith(".jpg"):
            user_id = file.split("_")[0]
            img_path = os.path.join("dataset", file)
            if not os.path.isfile(img_path):
                print(f"[SKIP] File not found: {img_path}")
                continue
            img = cv2.imread(img_path)

            try:
                embedding = DeepFace.represent(img_path=img_path, model_name='ArcFace', enforce_detection=True)[0]["embedding"]
                if not isinstance(embedding, list) or len(embedding) != 512:
                    print(f"[SKIP] Invalid embedding for {file}")
                    continue

                if user_id not in registered_faces:
                    registered_faces[user_id] = []
                registered_faces[user_id].append(embedding)
            except Exception as e:
                print(f"[ERROR] Embedding failed for {file}: {e}")

    averaged_faces = {uid: np.mean(embs, axis=0).tolist() for uid, embs in registered_faces.items()}
    with open("face_embeddings.pkl", "wb") as f:
        pickle.dump(averaged_faces, f)

    print(f"[INFO] Embeddings updated for {len(averaged_faces)} user(s).")


update_embeddings()

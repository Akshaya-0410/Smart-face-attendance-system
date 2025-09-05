import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide warnings/info

import tensorflow as tf
import cv2
import os
import sqlite3
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import time
from mtcnn import MTCNN
from deepface import DeepFace
import tensorflow as tf
from PIL import Image
import logging
tf.get_logger().setLevel(logging.ERROR)
import gc


import numpy as np
def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return np.dot(a, b)

# Config
SIMILARITY_THRESHOLD = 0.75
MIN_GAP = 0.05
DISTANCE_GAP_MIN = 1.0 
# Ensure dataset folder exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Load registered face embeddings
import pickle

def average_embeddings(embs):
    if not embs:
        return None
    avg = np.mean(embs, axis=0)
    return avg if avg.shape == (512,) else None

def load_registered_faces():
    if not os.path.exists("face_embeddings.pkl"):
        return {}

    with open("face_embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    cleaned_data = {}
    for uid, emb in data.items():
        if isinstance(emb, list):
            emb = np.array(emb)
        if isinstance(emb, np.ndarray):
            if emb.shape == (1, 512):
                emb = emb.reshape(512,)
            if emb.shape == (512,):
                cleaned_data[uid] = emb
            else:
                print(f"[❌] Skipping ID {uid}: Wrong shape {emb.shape}")
        else:
            print(f"[❌] Skipping ID {uid}: Not a numpy array")
    return cleaned_data




# Load and clean known face embeddings
raw_faces = load_registered_faces()
known_faces = {}

for uid, emb in raw_faces.items():
    if isinstance(emb, list):
        emb = np.array(emb)
    if isinstance(emb, np.ndarray):
        if emb.shape == (1, 512):
            emb = emb.reshape(512,)
        if emb.shape == (512,):
            known_faces[uid] = emb
        else:
            print(f"[❌] Skipping ID {uid}: Wrong shape {emb.shape}")
    else:
        print(f"[❌] Skipping ID {uid}: Not a numpy array")

print(f"[INFO] Loaded {len(known_faces)} valid face embeddings.")


#facenet_model = DeepFace.build_model("ArcFace")
from deepface import DeepFace
import numpy as np
import cv2

# Add this BEFORE take_attendance() function

def get_face_embedding(face_img):
    try:
        # face_img should be a BGR or RGB NumPy array of size (112, 112, 3)
        # DeepFace will handle conversion internally

        embedding_obj = DeepFace.represent(
            img_path=face_img, 
            model_name='ArcFace', 
            enforce_detection=False
        )
        embedding = embedding_obj[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"[❌] get_face_embedding ERROR: {e}")
        return None




def refresh_faces():
    global known_faces
    known_faces = load_registered_faces()
    messagebox.showinfo("Refreshed", "Face data has been reloaded.")

# Fetch name from database
def get_user_info(user_id):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users WHERE id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# Record attendance
def record_attendance(user_id, user_name, slot):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    date = time.strftime("%d-%B-%Y")
    time_now = time.strftime("%H:%M:%S")
    cursor.execute("SELECT * FROM attendance WHERE user_id=? AND date=? AND slot=?", (user_id, date, slot))
    existing = cursor.fetchone()
    if not existing:
        cursor.execute("INSERT INTO attendance (user_id, name, date, time, slot) VALUES (?, ?, ?, ?, ?)",
                       (user_id, user_name, date, time_now, slot))
        print(f"[Saved to DB] ID={user_id}, Name={user_name}, Slot={slot}, Time={time_now}")
        conn.commit()
    conn.close()
    return date, time_now

# UI setup
window = tk.Tk()
window.title("Smart Face Attendance - Tracking")
window.geometry("1280x720")
window.configure(bg="#C6DBEF")

tk.Label(window, text="Smart Face Attendance - Tracking", bg="#1C1C1C", fg="white",
         font=("Poppins", 24, "bold"), height=2).pack(fill="both")


attendance_frame = tk.Frame(window, bg="#9ECAE1", relief="ridge", borderwidth=3)
attendance_frame.place(relx=0.05, rely=0.2, relwidth=0.9, relheight=0.7)

tk.Label(attendance_frame, text="Attendance Tracking", bg="#DEEBF7", fg="black",
         font=("Poppins", 18, "bold")).pack(fill="both", pady=10)

slot_filter_var = tk.StringVar()
slot_filter = ttk.Combobox(attendance_frame, textvariable=slot_filter_var, font=("Poppins", 12), width=20)
slot_filter['values'] = ["All", "Morning", "Afternoon", "Evening"]
slot_filter.current(0)
slot_filter.pack(pady=10)

# Table
table_frame = tk.Frame(attendance_frame, bg="white", relief="ridge", borderwidth=2)
table_frame.pack(pady=10, padx=10, fill="both", expand=True)

columns = ("ID", "NAME", "DATE", "TIME", "SLOT")
tree = ttk.Treeview(table_frame, columns=columns, show="headings")
for col in columns:
    tree.heading(col, text=col, anchor="center")
tree.pack(fill="both", expand=True)

# Load attendance table
def load_attendance():
    for row in tree.get_children():
        tree.delete(row)
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    # Create attendance table if it doesn't exist
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
    selected_slot = slot_filter_var.get()
    if selected_slot == "All":
        cursor.execute("SELECT user_id, name, date, time, slot FROM attendance WHERE date=? ORDER BY time DESC", (today,))
    else:
        cursor.execute("SELECT user_id, name, date, time, slot FROM attendance WHERE date=? AND slot=? ORDER BY time DESC", (today, selected_slot))
    for row in cursor.fetchall():
        tree.insert("", "end", values=row)
    conn.close()

slot_filter.bind("<<ComboboxSelected>>", lambda e: load_attendance())

def reset_attendance():
    confirm = messagebox.askyesno("Confirm Reset", "Are you sure you want to delete ALL attendance records?")
    if confirm:
        try:
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM attendance")
            conn.commit()
            conn.close()
            load_attendance()
            messagebox.showinfo("Reset Complete", "✅ All attendance records have been deleted.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset attendance:\n{e}")

#known_faces = load_registered_faces()
# Webcam Attendance
def take_attendance():
    # Load and clean known face embeddings
    raw_faces = load_registered_faces()
    known_faces = {}

    for uid, emb in raw_faces.items():
        if isinstance(emb, list):
           emb = np.array(emb)
        if isinstance(emb, np.ndarray):
           if emb.shape == (1, 512):
               emb = emb.reshape(512,)
           if emb.shape == (512,):
               known_faces[uid] = emb
           else:
               print(f"[❌] Skipping ID {uid}: Wrong shape {emb.shape}")
        else:
            print(f"[❌] Skipping ID {uid}: Not a numpy array")

    print(f"[INFO] Loaded {len(known_faces)} valid face embeddings.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        messagebox.showerror("Webcam Error", "Webcam could not be opened.")
        return

    slot = simpledialog.askstring("Input Slot", "Enter slot (Morning, Afternoon, Evening):")
    if not slot:
        messagebox.showwarning("Slot Missing", "Slot is required.")
        return

    slot = slot.strip().capitalize()
    if slot not in ["Morning", "Afternoon", "Evening"]:
        messagebox.showerror("Invalid Slot", "Slot must be Morning, Afternoon, or Evening.")
        return

    detector = MTCNN()
    marked_ids = set()
    messagebox.showinfo("Instructions", "Look at the webcam. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        print(f"[INFO] Detected {len(faces)} faces.")

        for face in faces:
            try:
                x, y, w, h = face['box']
                x, y = max(x, 0), max(y, 0)
                face_crop = frame[y:y+h, x:x+w]
                face_crop = cv2.resize(face_crop, (112, 112))

                new_embedding = get_face_embedding(face_crop)
                if new_embedding is None or not isinstance(new_embedding, np.ndarray) or new_embedding.shape != (512,):
                   print(f"[❌] Invalid embedding. Skipping face.")
                   continue

                distances = []
                for user_id, known_emb in known_faces.items():
                    if known_emb is None or known_emb.shape != (512,):
                        continue
                    try:
                        sim = cosine_similarity(known_emb, new_embedding)
                        distances.append((user_id, sim))
                        print(f"[INFO] Similarity: {sim:.2f} with user {user_id}")
                    except Exception as e:
                        print(f"[Similarity ERROR] {e}")

                if not distances:
                    continue

                distances.sort(key=lambda x: x[1], reverse=True)
                best_match, best_sim = distances[0]
                second_best_sim = distances[1][1] if len(distances) > 1 else 0.0
                recognized_id = int(best_match)
                
                match_accepted = False
# Match threshold tuning
                THRESHOLD = 0.5
                GAP_THRESHOLD = 0.03
                if best_sim >= THRESHOLD and (best_sim - second_best_sim) >= GAP_THRESHOLD:
                    match_accepted = True
                    if recognized_id not in marked_ids:
                       user_name = get_user_info(recognized_id)
                       if user_name:
                            date, time_now = record_attendance(recognized_id, user_name, slot)
                            marked_ids.add(recognized_id)

                            # ✅ Update TreeView
                            already_in_ui = any(tree.item(child)['values'][0] == recognized_id for child in tree.get_children())
                            if not already_in_ui:
                                tree.insert("", "end", values=(recognized_id, user_name, date, time_now, slot))

                            print(f"[✅] Marked: {user_name} | Similarity: {best_sim:.2f}")
                       else:
                           print(f"[❌ NAME NOT FOUND] ID={recognized_id}")
                else:
                    print(f"[❌] Rejected match (ID={recognized_id}, Similarity={best_sim:.2f}, Gap={best_sim - second_best_sim:.2f})")

        # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, f"ID: {recognized_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            except Exception as e:
                print(f"[ERROR] Face processing error: {e}")


        cv2.imshow("Webcam Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    load_attendance()
    messagebox.showinfo("Done", f"Marked attendance for {len(marked_ids)} user(s).")

# Group Photo Attendance
def process_group_photo():
    # Load and clean known face embeddings
    raw_faces = load_registered_faces()
    known_faces = {}

    for uid, emb in raw_faces.items():
        try:
           if isinstance(emb, list):
              emb = np.array(emb)
           if isinstance(emb, np.ndarray):
              if emb.shape == (1, 512):
                 emb = emb.reshape(512,)
              if emb.shape == (512,):
                 known_faces[uid] = emb
              else:
                 print(f"[❌] Skipping ID {uid}: Wrong shape {emb.shape}")
           else:
               print(f"[❌] Skipping ID {uid}: Not a numpy array")
        except Exception as e:
            print(f"[ERROR] Cleaning embedding for ID {uid}: {e}")
    print(f"[INFO] Loaded {len(known_faces)} valid face embeddings.")
    file_path = filedialog.askopenfilename(
        title="Select Group Photo",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        return

    try:
        pil_image = Image.open(file_path).convert("RGB")
        image_np = np.array(pil_image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image:\n{e}")
        return

    # Resize if image is too large
    h, w = image_np.shape[:2]
    if w > 1400:
        scale = 1200 / w
        new_size = (1200, int(h * scale))
        image_np = cv2.resize(image_np, new_size)
        print(f"[INFO] Resized group image to {new_size[0]}x{new_size[1]}")

    # Ask for slot
    slot = simpledialog.askstring("Input Slot", "Enter slot (Morning, Afternoon, Evening):")
    if not slot:
        messagebox.showwarning("Slot Missing", "Slot is required.")
        return

    slot = slot.strip().capitalize()
    if slot not in ["Morning", "Afternoon", "Evening"]:
        messagebox.showerror("Invalid Slot", "Slot must be Morning, Afternoon, or Evening.")
        return

    # Detect faces using MTCNN
    detector = MTCNN()
    try:
        faces = detector.detect_faces(image_np)
    except Exception as e:
        messagebox.showerror("Detection Error", f"MTCNN failed: {e}")
        return

    print(f"[INFO] MTCNN detected {len(faces)} face(s)")
    if not faces:
        messagebox.showwarning("No Faces", "No faces detected in the group photo.")
        return

    marked_ids = set()

    for idx, face in enumerate(faces):
        try:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            x2, y2 = min(x + w, image_np.shape[1]), min(y + h, image_np.shape[0])
            face_crop = image_np[y:y2, x:x2]

            if face_crop.size == 0:
                print(f"[⚠️] Skipped empty crop at index {idx}")
                continue

            # Resize to 112x112 for ArcFace
            face_crop = cv2.resize(face_crop, (112, 112))

            # Get embedding
            new_embedding = get_face_embedding(face_crop)
            if new_embedding is None:
               print(f"[WARNING] Face {idx}: get_face_embedding returned None.")
               continue

            if isinstance(new_embedding, list):
               new_embedding = np.array(new_embedding)

            if not isinstance(new_embedding, np.ndarray) or new_embedding.shape != (512,):
               print(f"[❌] Face {idx}: Invalid embedding format or shape: {type(new_embedding)}, shape={getattr(new_embedding, 'shape', None)}")
               continue
 

            # Compare to known embeddings
            distances = []
            for user_id,avg_embedding in known_faces.items():
                if isinstance(avg_embedding,np.ndarray) and avg_embedding.shape == (512,):
                    sim = cosine_similarity(avg_embedding, new_embedding)
                    distances.append((user_id,sim))

            if not distances:
                print(f"[SKIP] No valid embeddings to compare for face {idx}.")
                continue

            distances.sort(key=lambda x: x[1], reverse=True)
            best_match, best_sim = distances[0]
            recognized_id = int(best_match)
            second_best_sim = distances[1][1] if len(distances) > 1 else 0.0

            # Recognition logic
            match_accepted = False
# Match threshold tuning
            THRESHOLD = 0.41
            GAP_THRESHOLD = 0.05
            if best_sim > THRESHOLD:
                match_accepted = True
                if recognized_id not in marked_ids:
                    user_name = get_user_info(recognized_id)
                    if user_name:
                        date, time_now = record_attendance(recognized_id, user_name, slot)
                        marked_ids.add(recognized_id)

                            # ✅ Update TreeView
                        already_in_ui = any(tree.item(child)['values'][0] == recognized_id for child in tree.get_children())
                        if not already_in_ui:
                            tree.insert("", "end", values=(recognized_id, user_name, date, time_now, slot))

                        print(f"[✅] Marked: {user_name} | Similarity: {best_sim:.2f}")
                    else:
                        print(f"[❌ NAME NOT FOUND] ID={recognized_id}")
            else:
                print(f"[❌] Rejected match (ID={recognized_id}, Similarity={best_sim:.2f}, Gap={best_sim - second_best_sim:.2f})")

            # Draw rectangle
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_np, f"ID: {recognized_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        except Exception as e:
            print(f"[ERROR] Failed to process face {idx}: {e}")

    # Show annotated image
    cv2.imshow("Group Faces", image_np)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    load_attendance()
    messagebox.showinfo("Done", f"Marked attendance for {len(marked_ids)} user(s).")

# Buttons
btn_frame = tk.Frame(attendance_frame, bg="#9ECAE1")
btn_frame.pack(pady=15)

tk.Button(btn_frame, text="Take Attendance", command=take_attendance,
          font=("Poppins", 12, "bold"), bg="black", fg="white", width=20).pack(side="left", padx=20)

tk.Button(btn_frame, text="Group Photo", command=process_group_photo,
          font=("Poppins", 12, "bold"), bg="black", fg="white", width=20).pack(side="left", padx=20)

tk.Button(btn_frame, text="Reset Attendance", command=reset_attendance,
          font=("Poppins", 12, "bold"), bg="green", fg="white", width=20).pack(side="left", padx=20)


tk.Button(btn_frame, text="Quit", command=window.quit,
          font=("Poppins", 12, "bold"), bg="red", fg="white", width=20).pack(side="left", padx=20)

load_attendance()
window.mainloop()

import os
import shutil

def remove_file(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)
        print(f"[INFO] Removed file: {filepath}")
    else:
        print(f"[SKIP] File not found: {filepath}")

def remove_folder(folderpath):
    if os.path.isdir(folderpath):
        shutil.rmtree(folderpath)
        print(f"[INFO] Removed folder: {folderpath}")
    else:
        print(f"[SKIP] Folder not found: {folderpath}")

if __name__ == "__main__":
    print("---- RESETTING SYSTEM ----")
    
    # Paths to delete
    files_to_delete = [
        "attendance.db",
        "face_embeddings.pkl",
        "arcface_embeddings.json",
        "attendance.csv"
    ]
    folders_to_delete = [
        "dataset",
        "registered_faces"
    ]
    
    for file in files_to_delete:
        remove_file(file)

    for folder in folders_to_delete:
        remove_folder(folder)

    print("[DONE] System reset completed. You can now re-register users.")

import os
import shutil

SOURCE_DIR = r"C:\Users\ADMIN\OneDrive\1 Đại Học\1_THESIS\AI Context"
DEST_DIR = r"C:\1_Local_Folder\Code\1_Thesis\Context"

def refresh_context():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        return

    # Create destination directory if it doesn't exist
    if not os.path.exists(DEST_DIR):
        print(f"Creating destination directory: {DEST_DIR}")
        os.makedirs(DEST_DIR)

    print(f"Copying contents from:\n  {SOURCE_DIR}\nto:\n  {DEST_DIR}\n")

    try:
        # Delete everything in the destination folder first
        for item in os.listdir(DEST_DIR):
            item_path = os.path.join(DEST_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        print("Cleared destination folder.")

        # Copy all contents
        shutil.copytree(SOURCE_DIR, DEST_DIR, dirs_exist_ok=True)
        print("Successfully refreshed context!")
    except Exception as e:
        print(f"An error occurred during copy: {e}")

if __name__ == "__main__":
    refresh_context()

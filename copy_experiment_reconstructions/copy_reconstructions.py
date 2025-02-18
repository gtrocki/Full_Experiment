import os
import shutil
import config


def copy_fused_files(source_path, destination_path):
    # Iterate over all folders in the source path
    for session_folder in os.listdir(source_path):
        session_folder_path = os.path.join(source_path, session_folder)

        # Check if it's a directory
        if os.path.isdir(session_folder_path):
            # Create a destination folder with the same name in the destination path
            destination_folder_path = os.path.join(destination_path, session_folder)
            os.makedirs(destination_folder_path, exist_ok=True)

            # Specify the source file and destination file paths within the "dense" folder
            dense_folder_path = os.path.join(session_folder_path, 'dense')
            source_file_path = os.path.join(dense_folder_path, 'fused.ply')
            destination_file_path = os.path.join(destination_folder_path, 'fused.ply')

            # Copy the 'fused.ply' file from the source to the destination folder
            try:
                shutil.copy(source_file_path, destination_file_path)
                print(f"File 'fused.ply' copied from {session_folder} to {destination_folder_path}")
            except FileNotFoundError:
                print(f"File 'fused.ply' not found in {session_folder}")


if __name__ == "__main__":

    source_path = config.SOURCE_PATH
    destination_path = config.DESTINATION_PATH

    copy_fused_files(source_path, destination_path)

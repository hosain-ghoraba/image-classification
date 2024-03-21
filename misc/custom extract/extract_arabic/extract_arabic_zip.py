import os
import shutil
import zipfile
import py7zr
from tqdm import tqdm
from tkinter import Button, Scrollbar, Text, Toplevel, filedialog
from tkinter import Tk
import tkinter.messagebox

def extract_zip_using_py7zr(source_path, destination_path):
    # delete the destination folder if it exists
    if os.path.exists(destination_path):
        print('Deleting the existing destination folder...')
        shutil.rmtree(destination_path)

    print("Extracting file : ", source_path, " to ", destination_path) 
    encoding_error_ignored = False
    with py7zr.SevenZipFile(source_path, mode='r') as zip_ref:
        try:
            zip_ref.extractall(path=destination_path)
        except UnicodeDecodeError:
            encoding_error_ignored = True

    print("Extraction complete.")
    return encoding_error_ignored

def extract_zip(source_path, destination_path):
    # delete the destination folder if it exists
    if os.path.exists(destination_path):
        print('Deleting the existing destination folder...')
        shutil.rmtree(destination_path)

    print("Extracting file : ", source_path, " to ", destination_path) 
    encoding_error_ignored = False
    with zipfile.ZipFile(source_path, 'r') as zip_ref:
        files = zip_ref.infolist()
        for file in tqdm(files, desc="Extracting files", unit="file"):
            try:
                file.filename = file.filename.encode('cp437', errors='ignore').decode('utf-8')  # try 'cp437' encoding first
            except UnicodeDecodeError:
                try:
                    file.filename = file.filename.encode('utf-8', errors='ignore').decode('utf-8')  # fallback to 'utf-8' if 'cp437' fails
                except UnicodeDecodeError:
                    encoding_error_ignored = True           
            zip_ref.extract(file, path=destination_path)

    print("Extraction complete.")
    return encoding_error_ignored
def show_error(message):
    # Create a new top-level window
    window = Toplevel()

    # Create a Text widget
    text = Text(window)
    text.insert('1.0', message)
    text.pack(side='left')

    # Create a Scrollbar widget
    scrollbar = Scrollbar(window, command=text.yview)
    scrollbar.pack(side='right', fill='y')

    # Link the Text widget and the Scrollbar widget
    text['yscrollcommand'] = scrollbar.set

    # Create a Button widget to close the window
    button = Button(window, text='OK', command=window.destroy)
    button.pack()
def choose_file_and_extract():
    # Create a file dialog for the user to choose a zip file
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[('Zip files', '*.zip')])

    if file_path:
        try:
            # If a file was selected, extract it to the same directory
            zip_file_name = os.path.basename(file_path)
            destination_path = os.path.join(os.path.dirname(file_path), os.path.splitext(zip_file_name)[0])
            encoding_error_ignored = extract_zip(file_path, destination_path)
            os.startfile(os.path.dirname(destination_path))
            if encoding_error_ignored:
                show_error("Some file names could not be decoded. Please check the extracted files.")
        except Exception as e:
            show_error(str(e))
            tkinter.messagebox.showerror("Error", str(e))
           

choose_file_and_extract()
import os
import shutil
import zipfile
from tkinter import Button, ttk ,filedialog, Label, TclError, Frame
from tkinterdnd2 import DND_FILES,TkinterDnD
import tkinter.messagebox



def extract_zip(source_path, destination_path, progressbar , drop_label):
    drop_label.config(text="Extracting....")
    drop_label.update()
    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)
    with zipfile.ZipFile(source_path, 'r') as zip_ref:
        files = zip_ref.infolist()
        progressbar['maximum'] = len(files)
        for i, file in enumerate(files):
            zip_ref.extract(file, path=destination_path)
            progressbar['value'] = i
            progressbar.update()
        progressbar['value'] = len(files)

        
        

def drop(event):
    file_path = event.data[1:-1]  # remove the curly braces
    if file_path:
        zip_file_name = os.path.basename(file_path)
        destination_path = os.path.join(os.path.dirname(file_path), os.path.splitext(zip_file_name)[0])
        extract_zip(file_path, destination_path, progressbar, drop_label)
        os.startfile(os.path.dirname(destination_path))

def choose_file_and_extract(root, progressbar, drop_label):
    file_path = filedialog.askopenfilename(filetypes=[('Zip files', '*.zip')])
    if file_path:
        zip_file_name = os.path.basename(file_path)
        destination_path = os.path.join(os.path.dirname(file_path), os.path.splitext(zip_file_name)[0])
        extract_zip(file_path, destination_path, progressbar, drop_label)
        os.startfile(os.path.dirname(destination_path))
        root.destroy()  # Close the window after execution


root = TkinterDnD.Tk()
root.title("Zip Extractor") 
# root.configure(bg='lightblue')
root.geometry('990x540')  # Set the window size to half of your screen size

progressbar = ttk.Progressbar(root, length=990, mode='determinate')
progressbar.pack() 

drop_label = Label(root, text="Drop your file here", font=("Arial", 24))
# drop_label.place(relx=0.5, rely=0.5, anchor='center')
drop_label.pack(pady=100)

choose_button = Button(root, text="Or Choose Manually", command=lambda: choose_file_and_extract(root, progressbar, drop_label), height=2, width=20, font=("Arial", 20), bg='green', fg='white', relief='raised', borderwidth=5)
choose_button.pack()

root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', drop)

try:
    root.mainloop()
except TclError:
    pass
except Exception as e:
    tkinter.messagebox.showerror("Error", str(e))


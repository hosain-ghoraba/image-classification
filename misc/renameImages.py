import os
import uuid
from PIL import Image
import warnings



# ----------------------------------------------------------------------------------------------------               
all_entities_path = './data set 2/'
all_entities_names = os.listdir(all_entities_path)
# ----------------------------------------------------------------------------------------------------
data_is_corrupted = False
print("\nVerifying all files are non-corrupted images...")
for entity_name in all_entities_names:
    entity_path = os.path.join(all_entities_path, entity_name)
    for filename in os.listdir(entity_path):
        file_path = os.path.join(entity_path, filename)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')  # treat warnings as exceptions within this context
                img = Image.open(file_path)  
                img.verify()
        except (IOError, SyntaxError, UserWarning) as e:  # catch UserWarning along with other exceptions
            print(type(e) , "for file", filename, ":", e)
            data_is_corrupted = True
            os.remove(file_path)  # delete the file
            print(f"file {filename} in folder {entity_name} is corrupted and has been deleted")     
print("your data is ok." if not data_is_corrupted else "done reporting error files.")  
    
# ----------------------------------------------------------------------------------------------------
# First pass to rename all files to a temporary unique name to avoid renaming a file with a name that belongs to another file in the same folder
print("giving temporary unique names...")
for entity_name in all_entities_names:
    entity_path = os.path.join(all_entities_path, entity_name)
    for filename in os.listdir(entity_path):
        temp_filename = str(uuid.uuid4()) + ".jpg"  # generate a unique filename
        source = os.path.join(entity_path, filename)
        destination = os.path.join(entity_path, temp_filename)
        os.rename(source, destination)
# ----------------------------------------------------------------------------------------------------
print("renaming...")
# then rename every file in every folder in the given path

for entity_name in all_entities_names:
    entity_path = os.path.join(all_entities_path, entity_name)
    i = 1
    for filename in os.listdir(entity_path):
        entity_name = entity_name.lower()
        new_filename = entity_name + '.' + str(i) + ".jpg"
        source = os.path.join(entity_path, filename)
        destination = os.path.join(entity_path, new_filename)
        os.rename(source, destination)  
        i += 1

print("done")        



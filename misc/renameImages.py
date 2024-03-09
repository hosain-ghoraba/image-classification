import os
import uuid
from PIL import Image

# ----------------------------------------------------------------------------------------------------               
all_entities_path = './data set 2/'
all_entities_names = os.listdir(all_entities_path)
# ----------------------------------------------------------------------------------------------------
data_is_corrupted = False
corrupted_files = []
print("\nvaryfiying all files are non-corrupted images...")
for entity_name in all_entities_names:
    entity_path = os.path.join(all_entities_path, entity_name)
    for filename in os.listdir(entity_path):
        try:
            img = Image.open(os.path.join(entity_path, filename))  # try to open the image
            img.verify()  # verify that it is, in fact, an image
        except (IOError, SyntaxError) as e:
            data_is_corrupted = True
            corrupted_files.append('folder : ' + entity_name + ' , file :' + filename)    
if data_is_corrupted:
    print("\nfound corrupted files, please fix them before proceeding :")
    print("\n".join(corrupted_files))
    exit()
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



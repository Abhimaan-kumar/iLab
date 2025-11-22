import os

# folder where images are stored
folder_path = r"dataset/training/Subconjunctival Hemorrage"   # change this

# new name pattern 
new_name = "image"  # files will become image_1.jpg, image_2.png etc

# supported image extensions
extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]

count = 1

for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    
    # skip folders
    if not os.path.isfile(file_path):
        continue
    
    # get extension
    _, ext = os.path.splitext(file)
    
    # check if it is an image
    if ext.lower() in extensions:
        new_file_name = f"im_{count}{ext}"
        new_file_path = os.path.join(folder_path, new_file_name)
        
        os.rename(file_path, new_file_path)
        count += 1

print("Renaming complete!")

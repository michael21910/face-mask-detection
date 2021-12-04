import pandas, os

name_format = "withMask"

img_dir = f"./../src/{name_format}Images/"

filename = f"./../src/{name_format}_filename.csv"

if (os.path.isfile(filename)):
    
    filenames = pandas.read_csv(filename)
    
    for index, row in filenames.iterrows():
        
        cur, new = os.path.isfile(img_dir + row['oldname']), os.path.isfile(img_dir + row['newname'])
        
        if cur and new:
            print(f"{index}: Both filenames are already taken.")
        elif cur:
            os.rename(img_dir + row['oldname'], img_dir + row['newname'])
        elif new:
            os.rename(img_dir + row['newname'], img_dir + row['oldname'])
        else:
            print(f"{index}: Neither filename is valid.")
            
else:
    
    print(f"File Error: {filename} does not exist.\n")
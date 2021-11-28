import os, pandas 

supported_images = [ ".jpg", ".png" ]

name_format = "withMask"

num_correct = 0 

_filename = f"./../src/{name_format}_filename.csv"

img_dir = f"./../src/{name_format}Images/"

def correct_name(filename):
    global name_format
    s_len, t_len, f_len = len(filename), len(name_format), 4
    return (((s_len - f_len) >= t_len) and (filename[:t_len] == name_format) and ((filename[t_len:s_len - f_len] == "") or (filename[t_len:s_len - f_len].isnumeric())))

def should_rename(filename):
    global supported_images, num_correct
    if (filename[-4:] in supported_images):
        if (correct_name(filename)):
            num_correct += 1
        else:
            return True
    return False

def save_to_csv():
    global dictionary, _filename
    FE = os.path.isfile(_filename)
    pandas.DataFrame(dictionary).to_csv(_filename, index = False, header = not FE, mode = ("a" if FE else "w"))

def update_names():
    global img_dir, _filename, dictionary

    nametracker = (
        ([ name["newname"] for _, name in pandas.read_csv(_filename).iterrows() ])
        if os.path.isfile(_filename) else []
    )

    for name in [ name for name in os.listdir(img_dir) ]:
        if correct_name(name) and not (name in nametracker):
            dictionary["newname"].append(name)
            dictionary["oldname"].append(f"not_{name}")

    save_to_csv()

    dictionary["newname"], dictionary["oldname"] = [], []

update_names()

filelist = [ name for name in os.listdir(img_dir) if should_rename(name) ]

dictionary = { "newname" : [], "oldname" : [] }

for index, name in enumerate(filelist):
    oldname, newname = name, name_format + str(num_correct + 1 + index) + name[-4:]
    dictionary["newname"].append(newname)
    dictionary["oldname"].append(oldname)
    os.rename(img_dir + oldname, img_dir + newname)

save_to_csv()
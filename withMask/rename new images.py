import os, pandas

supported_images = [ ".jpg", ".png" ]

name_format = "withMask"

num_correct = 0 

_filename = "filename.csv"

def correct_name(filename):
    global name_format
    return (name_format in filename)

def should_rename(filename):
    global supported_images, num_correct
    if (filename[-4:] in supported_images):
        if (correct_name(filename)):
            num_correct += 1
        else:
            return True
    return False

filelist = [ name for name in os.listdir("./") if should_rename(name) ]

dictionary = { "newname" : [], "oldname" : [] }

for index, name in enumerate(filelist):
    oldname, newname = name, name_format + str(num_correct + 1 + index) + name[-4:]
    dictionary["newname"].append(newname)
    dictionary["oldname"].append(oldname)
    os.rename(oldname, newname)

if (os.path.isfile(_filename)):
    pandas.DataFrame(dictionary).to_csv(_filename, mode = "a", header = False, index = False)
else:
    pandas.DataFrame(dictionary).to_csv(_filename, header = True, index = False)
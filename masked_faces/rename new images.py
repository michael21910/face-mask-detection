import os, pandas

supported_images = [ ".jpg", ".png" ]

num_correct = 0 

def correct_name(filename):
    return ("masked_faces_no" in filename)

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
    oldname, newname = name, "masked_faces_no" + str(num_correct + 1 + index) + name[-4:]
    dictionary["newname"].append(newname)
    dictionary["oldname"].append(oldname)
    os.rename(oldname, newname)
    
pandas.DataFrame(dictionary).to_csv("filename.csv", mode = "a", header = False, index = False)
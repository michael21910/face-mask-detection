import pandas, os
filenames = pandas.read_csv("filename.csv")
for index, row in filenames.iterrows():
    cur, new = os.path.isfile(row['oldname']), os.path.isfile(row['newname'])
    if cur and new:
        print(f"{index}: Both filenames are already taken.")
    elif cur:
        os.rename(row['oldname'], row['newname'])
    elif new:
        os.rename(row['newname'], row['oldname'])
    else:
        print(f"{index}: Neither filename is valid.")
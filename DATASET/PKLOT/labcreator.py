import os

F1 = "PUC"
F2 = "UFPR04"
F3 = "UFPR05"

def createlab(pth):
    res = ""
    try:
        dirs1 = os.listdir(pth)
    except NotADirectoryError as ndex:
        return
    for dir in dirs1:
        try:
            dirs2 = os.listdir(os.path.join(pth,dir))
            date = 0
        except NotADirectoryError as ndex:
            continue
        for dirdate in dirs2:
            try:
                date += 1
                if date < 2 and date > 4: continue
                else:
                    dirs3 = os.listdir(os.path.join(pth,dir,dirdate))
            except NotADirectoryError as ndex:
                continue
            for diroccp in dirs3:
                try:
                    dirs4 = os.listdir(os.path.join(pth,dir,dirdate,diroccp))
                except NotADirectoryError as ndex:
                    continue
                for imges in dirs4:
                    if diroccp == "Occupied": res += (pth+"/"+dir+"/"+dirdate+"/"+diroccp+"/"+imges+" 1\n")
                    else: res += (pth+"/"+dir+"/"+dirdate+"/"+diroccp+"/"+imges+" 0\n")
    res = res[:-1]
    with open("LABELS/"+pth+"_2DaysVal.txt", "x") as text_file:
        text_file.write(res)

createlab(F1)
createlab(F2)
createlab(F3)

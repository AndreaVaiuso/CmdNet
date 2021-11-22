import os
f = open("requirements.txt","r",encoding="utf-8")
content = f.read()
lines = content.splitlines()
res = {}
for line in lines:
	entry = line.split("==")
	res[entry[0]] = entry[1]

print(res)

f = open("requirements2.txt","r",encoding="utf-8")
content = f.read()
lines = content.splitlines()
for line in lines:
	entry = line.split("==")
	if entry[0] in res:
		continue
	else:
		print("---",entry[0])
		res[entry[0]] = entry[1]
result = ""
for key in res:
	result += key + "==" + res[key] + "\n"

f = open("requirements.txt","w",encoding="utf-8")
f.writelines(result)

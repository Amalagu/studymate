import math

myfile = open('courselist.txt', 'r').readlines()

newlist = []
def stringlist(thelist):
    text = ''
    for string in thelist:
        text += f'{str(string)} '
    return text
for line in myfile:
    newlist.append(line.split())

writelist = ['[',]
for line in newlist:
    course_code = line[0] + ' ' + line[1][:-1]
    course_title = stringlist(line[2:-3]) 
    unit = int(line[-3][1:]) + int(line[-2]) + int(line[-1][:-1])
    level = int(line[1][0])
    if int(line[1][2]) % 2 != 0:
        semester = "Harmattan"
    else:
        semester = "Rain"
    text = '{ ' + f'"semester": "{semester}", "code": "{course_code}", "title": "{course_title}", "unit": {unit}, "level": {level}' + ' },'
    print(text)
    writelist.append(text)

writelist.append(']')
print(len(writelist))
newfile = open("newfile.txt", 'a')
for line in writelist:
    newfile.writelines(line)
newfile.close()

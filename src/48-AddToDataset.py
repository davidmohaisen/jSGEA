from shutil import copyfile

f = open("/media/ahmed/HDD/ACSAC2021/graphs/malware/AV1_IoTReports.csv","r")
lines = f.readlines()[1:]
Needed = [1500,2738]
for line in lines:
    fileName,family = line.split(";")
    if "mirai" in family and Needed[0] != 0:
        try:
            Needed[0] -= 1
            copyfile("/media/ahmed/HDD/ACSAC2021/graphs/malware/graphs/"+fileName+".dot", "/home/ahmed/Documents/Projects/IOT-CFG-ATTACK-Journal/Dataset/Mirai/"+fileName+".dot")
        except:
            pass
    elif "tsunami" in family and Needed[1]!= 0:
        try:
            Needed[1] -= 1
            copyfile("/media/ahmed/HDD/ACSAC2021/graphs/malware/graphs/"+fileName+".dot", "/home/ahmed/Documents/Projects/IOT-CFG-ATTACK-Journal/Dataset/Tsunami/"+fileName+".dot")
        except:
            pass

#!/opt/local/bin/python3.3

from sys import argv

if (len(argv) == 4):
	script, sharedPath, designPath, outputPath = argv
else:
	print("ERROR: Too few/many arguments")
	print("Usage: convert-to-r-csv.py [shared-file] [design-file] [output-csv-file]")
	exit(-1)

designData = {}
with open(designPath) as designFile:
	for line in designFile:
		tmp = line.strip().split()
		key = designData.get(tmp[0].strip())
		if key == None:
			designData[tmp[0].strip()] = tmp[1].strip()
		else:
			print("Contains duplicate keys in design file, please recheck")
			exit(-1)
				
sharedData = []				
with open(sharedPath) as sharedFile:
	for line in sharedFile:
		tmp = [x.strip() for x in line.split()]
		sharedData.append(tmp)
	
with open(outputPath, mode="w") as outputFile:
	header = "group"
	for cell in sharedData[0][3:]:
		header +=  "," + str(cell)
	outputFile.write(header + "\n")

	for row in sharedData[1:]:
		line = str(designData[row[1]])
		for x in row[3:]:
			line +=  "," + str(x)
		outputFile.write(line + "\n")
	
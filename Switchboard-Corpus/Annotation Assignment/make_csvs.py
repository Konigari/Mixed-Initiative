import os
import pandas as pd 

saurabh = [ './saurabh/' + x for x in os.listdir('./saurabh') ] 

for files in saurabh:
	file = open(files,'r')
	name = files.split('/')[2]
	lines = file.readlines()
	data = []
	for line in lines:
		var = line.split("|")[:2]
		for x in range(3):
			var.append('')
		data.append(var)
	df = pd.DataFrame(data,columns=['Person','Conversation','Custom Tag','Custom Tag 2','Topic Description'])
	df.to_csv('./saurabh_csv/'+name[:4]+'.csv',index=False)	

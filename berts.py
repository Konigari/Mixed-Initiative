import mxnet as mx
from bert_embedding import BertEmbedding
import numpy as np
import os



def cos_sim(a, b):
	return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

# ctx = mx.gpu()
pos = 0
count = 0
posx = 0
countx = 0
lists = os.listdir('./glove_filter/processedanno/')
for name in lists:
	file = open('./glove_filter/processedanno/'+name,'r')
	lines = file.readlines()
	string = ''
	for line in lines:
		if len(line.split("|")) > 1:
			string+=line.split("|")[1]+"\n"


	bert_abstract = string.strip().split("\n")
	sentences = bert_abstract
	bert = BertEmbedding()

	result = bert(sentences)
	scan_range = 8
	print(file.name)
	for index, utterance in enumerate(lines):
		if index < scan_range or index >= len(lines) - scan_range:
			continue
		
		a1 = []
		a2 = []
		for i in range(-scan_range,0):
			sen1 = np.mean(result[index][1],axis=0)
			sen2 = np.mean(result[index+i][1],axis=0)
			a1.append(cos_sim(sen1, sen2))
		for i in range(1,scan_range+1):
			sen1 = np.mean(result[index][1],axis=0)
			sen2 = np.mean(result[index+1][1],axis=0)
			a2.append(cos_sim(sen1, sen2))
		a1 = np.array(a1)
		a2 = np.array(a2)
		arr1 = a1
		arr2 = a2
		a1 = a1[a1 > 0.55]
		a2 = a2[a2 > 0.55]
		if len(lines[index].split("|")) > 3:
			
			if "+" in lines[index].split("|")[3]:
				

				if len(a1)-len(a2) <= 0:
					sum1 = np.sum(arr1)
					sum2 = np.sum(arr2)
					val = np.exp(sum1-sum2)
					pos += 1
					if len(lines[index].split("|")) == 5:
						if val >= 0.5:
							if 's' in lines[index].split("|")[4]:
								posx += 1
								countx += 1
								print("Correct",lines[index],sum1-sum2)
							else:
								countx += 1
								print("Wrong=>sbd",lines[index],sum1-sum2)

						else:
							if 'i' in lines[index].split("|")[4]:
								posx += 1
								countx += 1
								print("Correct",lines[index],sum1-sum2)
								
							else:
								countx += 1
								print("wrong=>i",lines[index],sum1-sum2)

				count += 1

		

	print("Topic Shift",pos,count)
	print("Sub vs I",posx,countx)

	

print(pos/count)

print(posx,countx)
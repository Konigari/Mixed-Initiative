import pandas as pd
import glob
import ipdb
from sklearn.feature_extraction.text import TfidfVectorizer

for file in glob.glob("./Switchboard-Corpus/ASSIGNMENTS CSV/*.csv"):
	df = pd.read_csv(file)
	#ipdb.set_trace()
	corpus = df[:]["Conversation"]
	corpus = (str(x) for x in corpus)
	vectorizer = TfidfVectorizer()
	print (vectorizer)
	X = vectorizer.fit_transform(corpus)
	print(vectorizer.get_feature_names())
	print(X)
	break 
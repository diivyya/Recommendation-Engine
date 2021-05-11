import pandas as pd        #read csv files
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
df = pd.read_csv("movie_dataset.csv")   #read_csv method of pandas        
#print(df.columns) #read all the columns/features 



##Step 2: Select Features
features = ['keywords','cast','genres','director']

##Step 3: Create a column in DF which combines all selected features
for feature in features:
	df[feature] = df[feature].fillna('')    #go insides all the features and fill all AN with NULL

def combine_features(row):
	return row['keywords']+ " "+ row['cast'] + " "+ row['genres'] + " " + row['director']

df["combined_features"]= df.apply(combine_features, axis=1)

#print ("Combined Features:", df["combined_features"].head())  all Features giant matrix


##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()       #two methods in this class  fit and

count_matrix = cv.fit_transform(df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix) 

# given a movie in row we have to find all other movies similar to it
#      0    1     2     3  --  -- 
#   0  1   0.8   0.2   0.5
#   1       1    0.3   0.6                    this index we will find and return movies in descending order with its max corresponding value
#   2             1    0.1
#   |
#   |
randomlist = []
for i in range(0,10):
   n = random.randint(1,1000)
   randomlist.append(n)

for num in randomlist:
	print(get_title_from_index(num))

movie_user_likes = input("Choose a Movie which you like from the above Movies(Type Movie Name): ")

## Step 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))    #gives list of tuples, value with index

## Step 7: Get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

## Step 8: Print titles of first 20 movies
i=0
for movie in sorted_similar_movies:
	print (get_title_from_index(movie[0]))
	i=i+1
	if(i>20):
		break
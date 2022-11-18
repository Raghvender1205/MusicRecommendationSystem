import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from recommend import *

triplets = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata = 'https://static.turi.com/datasets/millionsong/song_data.csv'
song_df_a = pd.read_table(triplets, header=None)
song_df_a.columns = ['user_id', 'song_id', 'listen_count']

# Read song  metadata
song_df_b = pd.read_csv(songs_metadata)
# Merge the two dataframes above to create input dataframe for recommender systems
song_df1 = pd.merge(song_df_a, song_df_b.drop_duplicates(['song_id']), on="song_id", how="left")
# song_df1.head()
print("Total no of songs:", len(song_df1))
song_df1 = song_df1.head(10000)
# Merge song title and artist_name columns to make a new column
song_df1['song'] = song_df1['title'].map(str) + " - " + song_df1['artist_name']

# 
song_gr = song_df1.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_gr['listen_count'].sum()
song_gr['percentage']  = song_gr['listen_count'].div(grouped_sum)*100
song_gr.sort_values(['listen_count', 'song'], ascending = [0, 1])
u = song_df1['user_id'].unique()
print("The no. of unique users:", len(u))
train, test_data = train_test_split(song_df1, test_size=0.20, random_state=0)
# print(train.head(5))
pm = PopularityRecommendation()
pm.create_p(train, 'user_id', 'song')
user_id1 = u[5]
pm.recommend_p(user_id1)
user_id2 = u[8]
pm.recommend_p(user_id2)
is_model = ItemSimilarityRecommendation()
is_model.create_s(train, 'user_id', 'song')

#Print the songs for the user
user_id1 = u[5]
user_items1 = is_model.get_user_items(user_id1)
print("------------------------------------------------------------------------------------")
print("Songs played by first user %s:" % user_id1)
print("------------------------------------------------------------------------------------")

for user_item in user_items1:
    print(user_item)

print("----------------------------------------------------------------------")
print("Similar songs recommended for the first user:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend_s(user_id1)

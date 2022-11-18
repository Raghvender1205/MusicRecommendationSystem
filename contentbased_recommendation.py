from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data.csv")
feature_cols = ['acousticness', 'danceability', 'duration_ms', 'energy',
                'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
                'speechiness', 'tempo', 'time_signature', 'valence', ]

scaler = MinMaxScaler()
normalized_df = scaler.fit_transform(df[feature_cols])
# print(normalized_df[:2])
indices = pd.Series(df.index, index=df['song_title']).drop_duplicates()
cosine = cosine_similarity(normalized_df)

def recommend(song_title, model_type = cosine):
    index = indices[song_title]
    score = list(enumerate(model_type[indices[song_title]]))
    similarity_score = sorted(score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:11]
    top_songs_idx = [i[0] for i in similarity_score]
    top_songs = df['song_title'].iloc[top_songs_idx]

    return top_songs

if __name__ == '__main__':
    print('Recommended Songs: ')
    print(recommend('Parallel Lines', cosine).values)
import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Her bir veri düğümünü temsil etmek için bir sınıf tanımlanıyor.
class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

# Filmlerin verilerini bağlı listede saklamak için kullanılan sınıf.
class MovieLinkedList:
    def __init__(self):
        self.head = None

    # Yeni bir düğüm eklemek için kullanılan fonksiyon.
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    # Bağlı listedeki verileri çekmek için kullanılan fonksiyon.
    def fetch_data(self):
        data = []
        current = self.head
        while current:
            data.append(current.data) 
            current = current.next
        return data

# Verileri okuma
credits_df = pd.read_csv("movies/tmdb_5000_credits.csv")
movies_df = pd.read_csv("movies/tmdb_5000_movies.csv")

# Verileri birleştirme ve sütunları seçme
movies_df = movies_df.merge(credits_df, on='title')
movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Null değerleri düşürme
movies_df.dropna(inplace=True)

# Verileri işleme
linked_list = MovieLinkedList()
for index, row in movies_df.iterrows():
    movie_data = {
        'movie_id': row['movie_id'],
        'title': row['title'],
        'overview': row['overview'].split(),
        'genres': [i['name'] for i in ast.literal_eval(row['genres'])],
        'keywords': [i['name'] for i in ast.literal_eval(row['keywords'])],
        'cast': [i['name'] for i in ast.literal_eval(row['cast'])][:3],
        'crew': [i['name'] for i in ast.literal_eval(row['crew']) if i['job'] == 'Director'][:1]
    }
    linked_list.append(movie_data)

# Bağlı listedeki verileri DataFrame'e çevirme
movie_data_list = linked_list.fetch_data()
movies_df_linked_list = pd.DataFrame(movie_data_list)

# Özellik mühendisliği
movies_df_linked_list['tags'] = (
    movies_df_linked_list['overview'] +
    movies_df_linked_list['genres'] +
    movies_df_linked_list['keywords'] +
    movies_df_linked_list['cast'] +
    movies_df_linked_list['crew']
)

new_df = movies_df_linked_list[['movie_id', 'title', 'tags']]

# 'tags' sütunundaki kelimeleri birleştirme ve küçük harfe çevirme
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Vektöre çevirme
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Kök bulma (Stemming)
ps = PorterStemmer()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join([ps.stem(i) for i in x.split()]))

# Benzerlik hesaplama
similarity = cosine_similarity(vectors)

# Öneri fonksiyonu
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Örnek öneri
recommend('Avengers: Age of Ultron')
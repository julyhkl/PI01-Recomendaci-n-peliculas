
import pandas as pd
import numpy as np
import json
import locale
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors


df=pd.read_csv('movies_dataset.csv') #ruta 
df2=pd.read_csv('credits.csv')


df['id'] = df['id'].apply(lambda x: pd.to_numeric(x, errors='coerce')).astype('Int64')
df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')
df['id'] = pd.to_numeric(df['id'], errors='coerce').astype(float)
df = pd.merge(df, df2, on='id', how='outer', indicator=True)


df.head(2)


from pandas.io.json import json_normalize

# Desanidar la columna "belongs_to_collection"
df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x: {} if pd.isna(x) else eval(x))
df = pd.concat([df.drop('belongs_to_collection', axis=1), df['belongs_to_collection'].apply(pd.Series)], axis=1)

# Mostrar el resultado
df.head()


df['crew'] = df['crew'].apply(lambda x: eval(x) if pd.notnull(x) else [])
df['crew'] = [[item['name'] for item in i if item['job'] == 'Director'] for i in df['crew']] 

df['crew']


df['cast'] = df['cast'].apply(lambda x: eval(x) if pd.notnull(x) else [])
df['cast'] = [[item['name'] for item in i] for i in df['cast']] 

df['cast'] 


df['genres'] = df['genres'].apply(lambda x: eval(x) if pd.notnull(x) else [])
df['genres'] = [[item['name'] for item in i] for i in df['genres']] 

df['genres'] 


# Rellena los campos 'revenue' y 'budget' con 0
df['revenue'] = df['revenue'].fillna(0)
df['budget'] = df['budget'].fillna(0)


# Elimina filas con valores nulos en 'release_date'
df = df.dropna(subset=['release_date'])


# Convertir el campo 'release_date'
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce').dt.strftime('%Y-%m-%d')

# Crear la columna 'release_year'
df['release_year'] = pd.to_datetime(df['release_date']).dt.year


# Convertir las columnas 'revenue' y 'budget' a tipo numérico
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')

# Crear la columna del retorno
def calcular_retorno (fila):
    if fila['budget'] != 0:
        return fila['revenue'] / fila['budget']
    else:
        return 0

# Rellenar los valores nulos o faltantes con 0
df['return'] = df.apply(calcular_retorno, axis=1)


df.drop(axis=1, columns=['adult','homepage','imdb_id','original_language','original_title','poster_path','production_countries','runtime','spoken_languages','status','tagline','video','_merge','name','poster_path','backdrop_path',0], inplace=True)


df.to_csv('df_movies.csv', index=False)

# Aplicaciones


"""
La función recibe un valor tipo cadena con el valor de un mes y devuelve la cantidad de películas lanzadas en dicho mes
"""

def cantidad_filmaciones_mes(mes: str):
    df["release_date"] = pd.to_datetime(df["release_date"], errors='coerce')
    meses = {"enero": 1,
            "febrero": 2, 
            "marzo": 3, 
            "abril": 4, 
            "mayo": 5, 
            "junio": 6, 
            "julio": 7,
            "agosto": 8, 
            "septiembre": 9, 
            "octubre": 10, 
            "noviembre": 11, 
            "diciembre": 12}
    
    mes_numero = meses.get(mes.lower())
    contador = 0 
    
    for fecha in df["release_date"]:
        if pd.notnull(fecha) and fecha.month == mes_numero:
            contador += 1

    return {'mes': mes, 'cantidad': contador}


df['release_date'] = pd.to_datetime(df['release_date'])


"""
La función recibe un valor tipo cadena con el valor de un día y devuelve la cantidad de películas lanzadas dicho día
"""

def cantidad_filmaciones_dia(dia:str):
    df["release_date"] = pd.to_datetime(df["release_date"])
    dias_semana = {
    'lunes': 0, 
    'martes' : 1,
    'miercoles' : 2,
    'jueves' : 3,
    'viernes' : 4,
    'sabado' : 5,
    'domingo' :6}
    
    dia_numero = dias_semana.get(dia.lower())
    contador = 0 
    
    for fecha in df["release_date"]:
        if fecha.weekday() == dia_numero:
            contador += 1
    return {'dia':dia, 'cantidad':contador}


"""
La función recibe un valor tipo cadena indicando el nombre de la película y devuelve el nombre de la película, año en el que se estrenó y popularidad de la misma.
En caso de no encontrar la película devuelve None
"""

def score_titulo(titulo_de_la_filmacion):
    # Filtrar el DataFrame por el título de la filmación
    pelicula = df[df['title'] == titulo_de_la_filmacion]
    if len(pelicula) > 0:
        # Obtener el año de estreno y la popularidad
        año_estreno = pelicula['release_year'].values[0]
        popularidad = pelicula['popularity'].values[0]

        return {'titulo': titulo_de_la_filmacion, 'anio':int(año_estreno), 'popularidad': round(popularidad,2)}
    else:
        return {'titulo': None, 'anio':None, 'popularidad': None}




def votos_promedio_titulo(titulo_de_la_filmacion):
    # Filtrar el DataFrame por el título de la filmación
    pelicula = df[df['title'] == titulo_de_la_filmacion]

    # Obtener la cantidad de votos y el valor promedio de las votaciones
    votos = pelicula['vote_count'].values[0]
    promedio_votos = pelicula['vote_average'].values[0]
    año_estreno = pelicula['release_year'].values[0]
    
    if votos >= 2000:
        return {'titulo': titulo_de_la_filmacion, 'anio':int(año_estreno), 'voto_total': int(votos), 'voto_promedio': promedio_votos}
    else:
        return {'titulo': None, 'anio':None, 'voto_total': None, 'voto_promedio': None }



"""
La función recibe un valor tipo cadena con el nombre del actor/actriz a buscar y devuelve la cantidad de films que hizo, el retorno total de todos los films que hizo y el promedio
de su retorno total. En caso de no encontrar al actor/actriz devuelve None
"""

def get_actor(nombre_actor: str):
    actor_films = df[df['cast'].apply(lambda x: nombre_actor in x)]
    if actor_films.empty:
        return {"mensaje": "El actor no fue encontrado en ninguna filmación."} 
    
    cantidad_films = actor_films.shape[0]
    retorno_total = actor_films['return'].sum()
    promedio_retorno = actor_films['return'].mean()
    
    return {"actor": nombre_actor, "cantidad_films": cantidad_films, "retorno_total": round(retorno_total,2), "promedio_retorno": round(promedio_retorno,2)}




"""
La función recibe un valor tipo cadena con el nombre del director/directora a buscar y devuelve el éxito del mismo a través del retorno. También devuelve cada película
que hizo, fecha de lanzamiento, costo y ganancia
"""

def get_director(nombre_director: str):
    director_data = df[df['crew'].apply(lambda x: nombre_director in x)]
    if director_data.empty:
        return {"mensaje": "Director no encontrado"}
    
    peliculas = []
    retorno_maximo = 0
    exito = None
    
    for i, row in director_data.iterrows():
        pelicula = {
            "titulo": row['title'],
            "fecha_lanzamiento": row['release_date'],
            "retorno": row['return'],
            "costo": row['budget'],
            "ganancia": row['revenue']
        }
        peliculas.append(pelicula)
        
        if row['return'] > retorno_maximo:
            retorno_maximo = row['return']
            exito = row['title']
    
    return {
        "director": nombre_director,
        "exito": exito,
        "peliculas": peliculas}



# Recomendación


# Converte las columnas a float vote_avarege
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')

# Ordenama por vote_average y toma las primeras 5000 filas
df_highly_rated = df.sort_values(by='vote_average', ascending=False).head(5000).reset_index(drop=True)


# Cosine_similarity es una función que calcula la similitud del coseno. La similitud del coseno es una metrica utilizada para determinar cuan similares son dos vectores.
#HashingVectorizer es una clase que convierte una coleccion de documentos de texto en una matriz de ocurrencias de tokens. 

# Aseguramos que los datos de la columna 'overview', 'genres' y 'production_companies' sean strings
df_highly_rated['overview'] = df_highly_rated['overview'].fillna('').astype('str')
df_highly_rated['genres'] = df_highly_rated['genres'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else '')
# Reemplazar los valores nulos con cadenas vacias
df_highly_rated['production_companies'] = df_highly_rated['production_companies'].fillna('')
df_highly_rated['production_companies'] = df_highly_rated['production_companies'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)

# Creamos una nueva columna llamada 'combined' que es una combinacion de las columnas 'overview', 'genres' y 'production_companies'. Esta columna se usara para calcular
# la similitud entre diferentes peliculas.
df_highly_rated['combined'] = df_highly_rated['overview'] + ' ' + df_highly_rated['genres'] + ' ' + df_highly_rated['production_companies']

# Convertimos todos los textos a minusculas para evitar duplicados
df_highly_rated['combined'] = df_highly_rated['combined'].str.lower()

# Inicializamos el HashingVectorizer
hash_vectorizer = HashingVectorizer(stop_words='english', n_features=2000)
# De esta manera evitamos que las palabras mas comunes afecten a nuestro procesamiento de datos y evitamos que se generen vectores mas grandes

# Aprende el vocabulario de 'combined' y transforma 'combined' en una matriz de vectores
hash_matrix = hash_vectorizer.fit_transform(df_highly_rated['combined'])

# Calculamos la similitud del coseno
cosine_sim = cosine_similarity(hash_matrix)

# Creamos un indice con los titulos de las peliculas
indices = pd.Series(df_highly_rated.index, index=df_highly_rated['title']).drop_duplicates()


df_highly_rated.to_csv('movies_final_combined.csv', index=False)


df_highly_rated = pd.read_csv('movies_final_combined.csv')

# En esta matriz, cada fila representa una película y cada columna representa un termino en las caracteristicas combinadas
cv = CountVectorizer(stop_words='english', max_features=5000)
count_matrix = cv.fit_transform(df_highly_rated['combined'])

# Creamos un modelo para encontrar los vecinos mas cercanos en un espacio de caracterisicaa
nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(count_matrix)

# Creamos un indice de titulos de peliculas y eliminamos los duplicados
indices = pd.Series(df_highly_rated.index, index=df_highly_rated['title']).drop_duplicates()


"""
La función recibe un valor tipo cadena con el nombre de la película a buscar y devuelve 5 películas similares. En caso de que 
""" 

def recomendacion(title):

    if title not in df_highly_rated['title'].values:
        return 'La pelicula no se encuentra en el conjunto de la base de datos.'
    else:
        index = indices[title]

        # Obtiene las puntuaciones de similitud de las 5 peliculas más cercanas
        distances, indices_knn = nn.kneighbors(count_matrix[index], n_neighbors=6)

        # Obtiene los indices de las peliculas y se omite el primer indice que es la pelicula misma
        movie_indices = indices_knn[0][1:]

        # Devuelve las 5 peliculas mas similares
        return df_highly_rated['title'].iloc[movie_indices].tolist()




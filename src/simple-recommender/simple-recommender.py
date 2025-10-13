import os
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import lancedb
from pydantic import ConfigDict
from lancedb.pydantic import vector, LanceModel

DATA_DIR = "./ml-latest-small"
DB_PATH = os.path.expanduser("~/.lancedb")
TABLE_NAME = "movielens_small"
EMBED_DIM = 64


def load_movielens(data_dir: str):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    links = pd.read_csv(os.path.join(data_dir, "links.csv"))
    return ratings, movies, links


def build_ratings_matrix(ratings: pd.DataFrame):
    return ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0.0)


def l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def compute_embeddings(reviewmatrix: pd.DataFrame, embed_dim: int) -> np.ndarray:
    matrix = reviewmatrix.values
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    embeddings_full = vh.T
    d = min(embed_dim, embeddings_full.shape[1])
    embeddings = embeddings_full[:, :d]
    return l2_normalize_rows(embeddings)


def escape_for_filter(value: str) -> str:
    return value.replace('"', '\\"')


def build_lancedb_table(reviewmatrix: pd.DataFrame,
                        movies: pd.DataFrame,
                        links: pd.DataFrame,
                        embeddings: np.ndarray):
    movies_idx = movies.set_index("movieId").reindex(reviewmatrix.columns).fillna({"genres": "", "title": ""})
    links_idx = links.set_index("movieId").reindex(reviewmatrix.columns).fillna({"imdbId": 0})

    num_movies = len(reviewmatrix.columns)
    if embeddings.shape[0] != num_movies:
        raise ValueError(f"Embeddings rows ({embeddings.shape[0]}) do not match number of movies ({num_movies}).")

    vector_dim = embeddings.shape[1]

    class Content(LanceModel):
        model_config = ConfigDict(protected_namespaces=())
        movie_id: int
        vector: vector(vector_dim)
        genres: str
        title: str
        imdb_id: int

        @property
        def imdb_url(self) -> str:
            iid = int(self.imdb_id or 0)
            if not iid:
                return ""
            return f"https://www.imdb.com/title/tt{iid:07d}"

    values = list(zip(
        reviewmatrix.columns.astype(int),
        [row.astype(float).tolist() for row in embeddings],
        movies_idx["genres"].astype(str).tolist(),
        movies_idx["title"].astype(str).tolist(),
        links_idx["imdbId"].fillna(0).astype(int).tolist(),
    ))
    keys = ["movie_id", "vector", "genres", "title", "imdb_id"]
    data = [dict(zip(keys, v)) for v in values]

    db = lancedb.connect(DB_PATH)
    db.drop_table(TABLE_NAME, ignore_missing=True)
    table = db.create_table(TABLE_NAME, data=pa.Table.from_pylist(data, schema=Content.to_arrow_schema()))
    try:
        table.create_index(column="vector", metric="cosine", num_partitions=1, num_sub_vectors=16)
    except Exception:
        pass
    return db, table, Content


def get_recommendations(table, title: str, top_k: int = 5):
    filt = f'title = "{escape_for_filter(title)}"'
    qvec_arr = table.to_lance().to_table(filter=filt)["vector"].to_numpy()
    if len(qvec_arr) == 0:
        raise ValueError(f"Title not found: {title}")
    query_vector = qvec_arr[0]
    results_df = table.search(query_vector).metric("cosine").limit(top_k + 1).to_pandas()
    results_df = results_df[results_df["title"] != title].head(top_k)
    out = []
    for _, row in results_df.iterrows():
        imdb_id = int(row.get("imdb_id", 0) or 0)
        imdb_url = f"https://www.imdb.com/title/tt{imdb_id:07d}" if imdb_id else ""
        out.append((int(row["movie_id"]), str(row["title"]), imdb_url))
    return out


def suggest_titles(movies: pd.DataFrame, query: str, limit: int = 10):
    if not query:
        return []
    q = query.lower()
    s = movies[movies["title"].str.lower().str.contains(q, na=False)]["title"].head(limit)
    return s.tolist()


def main():
    parser = argparse.ArgumentParser(description="Simple movie recommender with LanceDB and SVD embeddings")
    parser.add_argument("--data_dir", default=DATA_DIR, help="Path to MovieLens ml-latest-small directory")
    parser.add_argument("--title", default=None, help="Movie title to query (if omitted, a prompt appears)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--embed_dim", type=int, default=EMBED_DIM, help="Embedding dimensionality (SVD components)")
    args = parser.parse_args()

    if not args.title:
        try:
            args.title = input("Enter movie title: ").strip()
        except EOFError:
            args.title = None
    if not args.title:
        print("No title provided.")
        return

    ratings, movies, links = load_movielens(args.data_dir)
    reviewmatrix = build_ratings_matrix(ratings)
    embeddings = compute_embeddings(reviewmatrix, args.embed_dim)
    _, table, _ = build_lancedb_table(reviewmatrix, movies, links, embeddings)

    try:
        recs = get_recommendations(table, args.title, top_k=args.top_k)
        for r in recs:
            print(r)
    except ValueError as e:
        print(str(e))
        suggestions = suggest_titles(movies, args.title)
        if suggestions:
            print("Did you mean:")
            for s in suggestions:
                print(f"- {s}")


if __name__ == "__main__":
    main()

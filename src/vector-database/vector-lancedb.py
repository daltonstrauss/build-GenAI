import os
import numpy as np
import lancedb
from lancedb.pydantic import vector, LanceModel
from lance.vector import vec_to_table


class CatsAndDogs(LanceModel):
    # Silence Pydantic's protected namespace warning
    model_config = {"protected_namespaces": ()}

    vector: vector(2)
    species: str
    breed: str
    weight: float


def main():
    db_path = os.path.expanduser("~/.lancedb")
    table_name = "cats_and_dogs"

    db = lancedb.connect(db_path)
    db.drop_table(table_name, ignore_missing=True)
    table = db.create_table(table_name, schema=CatsAndDogs)

    # Seed data
    data = [
        CatsAndDogs(vector=[1.0, 0.0], species="cat", breed="shorthair", weight=12.0),
        CatsAndDogs(vector=[-1.0, 0.0], species="cat", breed="himalayan", weight=9.5),
        CatsAndDogs(vector=[0.0, 10.0], species="dog", breed="samoyed", weight=47.5),
        CatsAndDogs(vector=[0.0, -1.0], species="dog", breed="corgi", weight=26.0),
    ]
    table.add([dict(d) for d in data])

    print(table.head().to_pandas())

    # Query vector dimension must match the schema's vector dim (2)
    query = np.array([0.25, 0.75], dtype=np.float32)

    # Use to_pandas() instead of deprecated to_df()
    results = table.search(query).limit(10).to_pandas()
    print(results)
    
    mat = np.random.randn(100_000, 16)
    table_name = "exercise3_ann"
    db.drop_table(table_name, ignore_missing=True)
    table = db.create_table(table_name, vec_to_table(mat))
    query = np.random.randn(16)
    results = table.search(query).limit(10).to_pandas()
    print(results)
    table.create_index(num_partitions=16, num_sub_vectors=8)
    results = table.search(query).limit(10).to_pandas()
    print(results)
    table.list_versions()

if __name__ == "__main__":
    main()
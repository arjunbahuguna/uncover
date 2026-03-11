import faiss
import numpy as np
import sqlite3


class MetadataManager:
    def __init__(self, db_path="vi_project.db"):
        # self.conn = sqlite3.connect(db_path)
        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
        # Create tracks table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracks (
                faiss_id INTEGER PRIMARY KEY,
                track_id TEXT NOT NULL,
                clique_id TEXT,
                dataset TEXT,
                file_path TEXT
            )
        ''')
        # Create index on clique_id
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_clique ON tracks(clique_id)')
        self.conn.commit()

    def insert_batch(self, data_list):
        query = 'INSERT INTO tracks VALUES (?, ?, ?, ?, ?)'
        self.cursor.executemany(query, data_list)
        self.conn.commit()

    def get_info_by_faiss_id(self, faiss_id):
        self.cursor.execute('SELECT * FROM tracks WHERE faiss_id = ?', (int(faiss_id),))
        return self.cursor.fetchone()


class MusicRetrievalSystem:
    def __init__(self, dimension=512, db_path="vi_project.db"):
        self.dimension = dimension
        self.db_manager = MetadataManager(db_path)
        # Initialize FAISS index with Inner Product
        self.index = faiss.IndexFlatIP(dimension)

    def add_song(self, vector, track_id, clique_id, dataset="Da-Tacos", file_path=""):
        vec = np.array(vector).astype('float32').reshape(1, -1)
        faiss.normalize_L2(vec)
        current_id = self.index.ntotal
        self.index.add(vec)
        self.db_manager.insert_batch([(current_id, track_id, clique_id, dataset, file_path)])

    def search(self, query_vector, k=5, return_vectors=False):
        q_vec = np.array(query_vector).astype('float32').reshape(1, -1)
        faiss.normalize_L2(q_vec)
        distances, indices = self.index.search(q_vec, k)

        results = []
        for i in range(k):
            idx = int(indices[0][i])
            if idx == -1:
                continue

            metadata = self.db_manager.get_info_by_faiss_id(idx)
            original_vec = self.index.reconstruct(idx) if return_vectors else None

            results.append({
                "score": float(distances[0][i]),
                "metadata": metadata,
                "vector": original_vec
            })
        return results


if __name__ == "__main__":
    system = MusicRetrievalSystem(dimension=512)

    # Add test song
    v = np.random.random(512)
    system.add_song(v, "Test_ID_001", "Clique_A")

    # Search
    res = system.search(v, k=1, return_vectors=True)
    print(f"Matched Track ID: {res[0]['metadata'][1]}")
    print(f"Vector Returned: {res[0]['vector'] is not None}")
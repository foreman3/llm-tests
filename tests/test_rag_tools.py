import unittest
import os
import pandas as pd
from rag_tools import rag_tools

class TestRAGTools(unittest.TestCase):

    def test_chunk_text_from_file(self):
        file_path = "./tests/sample.txt"
        file_path.write_text("One. Two. Three.")
        rt = rag_tools()
        df = rt.chunk_text(str(file_path), chunk_size=10)
        assert len(df) == 2
        assert list(df.columns) == ["chunk_id", "text"]


    def test_store_and_query(self):
        text = "Alpha one. Beta two. Gamma three."
        store_path = "./tests/store.pkl"
        rt = rag_tools()
        rt.store_chunks(text, str(store_path), chunk_size=12)
        assert store_path.exists()
        assert os.path.exists(str(store_path) + ".csv")

        result = rt.query_store("Beta two.", str(store_path), k=1)
        assert len(result) == 1
        assert "Beta two" in result.iloc[0]["text"]

if __name__ == "__main__":
    unittest.main()

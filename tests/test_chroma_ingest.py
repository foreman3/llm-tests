import unittest
import tempfile
from pathlib import Path

from chroma_ingest import ChromaIngestPipeline


class TestChromaIngest(unittest.TestCase):
    def test_ingest_and_query(self):
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            data_dir.mkdir()
            file_path = data_dir / "sample.txt"
            file_path.write_text("Alpha one. Beta two.")

            pipeline = ChromaIngestPipeline()
            store_path = Path(td) / "store"
            pipeline.ingest_folder(str(data_dir), str(store_path), chunk_size=12)

            self.assertTrue((store_path / "chroma.sqlite3").exists())
            self.assertTrue(Path(str(store_path) + ".csv").exists())

            result = pipeline.query_store("Beta two", str(store_path), k=1)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["filename"], "sample.txt")


if __name__ == "__main__":
    unittest.main()

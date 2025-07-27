import unittest
from pathlib import Path
import shutil
import gc
import time

from chroma_ingest import ChromaIngestPipeline

class TestChromaIngest(unittest.TestCase):
    def test_ingest_and_query(self):
        # Setup paths
        temp_dir = Path("./temp")
        store_path = temp_dir / "store"
        data_dir = temp_dir / "data"

        # Clean up any previous test artifacts
        if store_path.exists():
            shutil.rmtree(store_path)
        if data_dir.exists():
            shutil.rmtree(data_dir)
        if temp_dir.exists():
            for f in temp_dir.glob("*.csv"):
                f.unlink()
        else:
            temp_dir.mkdir(parents=True, exist_ok=True)

        data_dir.mkdir(parents=True, exist_ok=True)
        file_path = data_dir / "sample.txt"
        file_path.write_text("Alpha one. Beta two.")

        pipeline = ChromaIngestPipeline()
        pipeline.ingest_folder(str(data_dir), str(store_path), chunk_size=12)

        self.assertTrue((store_path / "chroma.sqlite3").exists())
        self.assertTrue(Path(str(store_path) + ".csv").exists())

        result = pipeline.query_store("Beta two", str(store_path), k=1)
        print("Chunk content:", result.iloc[0]["text"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["filename"], "sample.txt")


if __name__ == "__main__":
    unittest.main()
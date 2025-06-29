import requests

from knowledge_server import run_server
from knowledge_ingest import KnowledgeIngestor


def test_ingest_text(tmp_path):
    text_file = tmp_path / "facts.txt"
    text_file.write_text("A cat is a animal. The animal needs water.")

    server, thread = run_server(host="localhost", port=0)
    url = f"http://{server.server_address[0]}:{server.server_address[1]}"

    ingestor = KnowledgeIngestor(server_url=url)
    ingestor.ingest(str(text_file))

    resp = requests.post(
        f"{url}/query", json={"text": "What does a cat need?"}, timeout=5
    )
    assert resp.json()["answer"] == "water"

    server.shutdown()
    thread.join()

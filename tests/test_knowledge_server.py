import json
import threading
from http.server import HTTPServer

import requests

from knowledge_server import KnowledgeStore, run_server


def start_test_server(path=None):
    server, thread = run_server(host="localhost", port=0, path=path)
    return server, thread


def stop_test_server(server: HTTPServer, thread: threading.Thread):
    server.shutdown()
    thread.join()


def test_add_and_query():
    server, thread = start_test_server()
    url = f"http://{server.server_address[0]}:{server.server_address[1]}"

    resp = requests.get(f"{url}/tools", timeout=5)
    assert resp.status_code == 200
    assert "query" in resp.json()

    data = {"subject": "dog", "predicate": "isa", "object": "animal"}
    resp = requests.post(f"{url}/add", json=data, timeout=5)
    assert resp.json().get("status") == "ok"

    data = {"subject": "animal", "predicate": "needs", "object": "food"}
    requests.post(f"{url}/add", json=data, timeout=5)

    resp = requests.post(
        f"{url}/query", json={"text": "What does a dog need?"}, timeout=5
    )
    assert resp.json().get("answer") == "food"

    stop_test_server(server, thread)

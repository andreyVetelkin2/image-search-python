import requests

def test_search():
    with open("sample.jpg", "rb") as f:
        r = requests.post("http://localhost:8000/search", files={"file": f})
        assert r.status_code == 200
        assert "results" in r.json()

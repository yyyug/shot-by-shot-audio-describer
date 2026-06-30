from app import app

def test_index_contains_form():
    client = app.test_client()
    response = client.get('/')
    assert b'<form' in response.data or b'upload' in response.data.lower()

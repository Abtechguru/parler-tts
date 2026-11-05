"""No GPU/model download required for these API contract tests."""
import os
from fastapi.testclient import TestClient
from api_service import server
client=TestClient(server.app)

def test_health():
    r=client.get('/health')
    assert r.status_code == 200
    assert r.json()['service']=='parler-mini'

def test_synthesis_requires_key():
    os.environ['PARLER_TTS_API_KEY']='unit-test-secret'
    r=client.post('/synthesize',json={'text':'Hello exam host','style':'host'})
    assert r.status_code==401

def test_bad_style_rejected():
    os.environ['PARLER_TTS_API_KEY']='unit-test-secret'
    r=client.post('/synthesize',headers={'X-API-Key':'unit-test-secret'},json={'text':'Hello exam host','style':'invalid'})
    assert r.status_code==422

def test_generation_mock(monkeypatch):
    os.environ['PARLER_TTS_API_KEY']='unit-test-secret'
    monkeypatch.setattr(server,'generate_wav',lambda text,style:b'RIFF'+b'0'*64)
    r=client.post('/synthesize',headers={'X-API-Key':'unit-test-secret'},json={'text':'Welcome to Duel Lab','style':'energy'})
    assert r.status_code==200
    assert r.headers['content-type']=='audio/wav'
    assert r.content.startswith(b'RIFF')
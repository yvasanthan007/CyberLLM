# CyberLLM — Defensive Cybersecurity Assistant

CyberLLM is a defensive cybersecurity project combining phishing URL
classification with searchable cybersecurity knowledge.

## Features

- **Phishing URL assessment:** Uses a trained scikit-learn model to classify
  submitted HTTP/HTTPS URLs as PHISHING or LEGITIMATE.
- **Cybersecurity knowledge retrieval:** Searches a local knowledge base
  containing 2,427 documents using lexical ranking.
- **REST API:** FastAPI endpoints with interactive Swagger documentation.
- **Input validation:** Rejects malformed URL inputs.
- **Health monitoring:** Reports whether the model and knowledge base load.

## Technology

- Python
- FastAPI and Uvicorn
- scikit-learn and joblib
- HTTPX
- Pytest
- MITRE ATT&CK knowledge sources

## Prerequisites

Use a compatible Python environment and install the dependencies listed
in `requirements.txt`.

## Run locally on Windows

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8765
```

Open the interactive API documentation:

http://127.0.0.1:8765/docs

Keep the server terminal running while sending API requests.

## API endpoints

### Health check

`GET /health`

Reports API status, model availability, and knowledge-base availability.

### Phishing URL assessment

`POST /phishing/predict`

Example request:

```json
{
  "url": "https://www.google.com/"
}
```

The response includes a prediction and a phishing score.

### Knowledge search

`POST /knowledge/search`

Example request:

```json
{
  "query": "phishing attack detection",
  "top_k": 3
}
```

Returns ranked knowledge-base documents. This endpoint currently performs
lexical retrieval; it does not generate LLM answers.

## Run tests

```powershell
python -m pytest -q
```

## Security and limitations

- Predictions are model assessments, not guarantees of safety.
- False positives and false negatives are possible.
- The phishing score is not a calibrated probability.
- Search results are lexical matches, not generated answers.
- A URL classification alone does not establish that a website is safe.
- Use the platform for defensive and educational purposes.

## Current verification

The API health check, clean URL prediction requests, knowledge search,
and invalid URL validation have been exercised successfully.
The existing test suite passed 6 tests in the recorded run.

Run the tests again against your current checkout before relying on this
status for a release.

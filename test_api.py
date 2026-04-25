import pytest
from fastapi.testclient import TestClient
from tanu import app  # make sure your file name is tanu.py

client = TestClient(app)


# -----------------------------
# Helper Functions
# -----------------------------
def register_user():
    response = client.post("/auth/register", json={
        "full_name": "Test User",
        "email": "testuser@example.com",
        "password": "password123"
    })
    return response


def login_user():
    response = client.post(
        "/auth/login",
        data={
            "username": "testuser@example.com",
            "password": "password123"
        }
    )
    return response


# -----------------------------
# TEST CASES
# -----------------------------

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_register_and_login():
    register_user()

    res = login_user()
    assert res.status_code == 200
    assert "access_token" in res.json()


def test_create_interview():
    login = login_user().json()
    token = login["access_token"]

    res = client.post(
        "/interviews",
        json={"role": "Python"},
        headers={"Authorization": f"Bearer {token}"}
    )

    assert res.status_code == 200
    assert "interview_id" in res.json()


def test_full_interview_flow():
    login = login_user().json()
    token = login["access_token"]

    # Create Interview
    res = client.post(
        "/interviews",
        json={"role": "Python"},
        headers={"Authorization": f"Bearer {token}"}
    )
    interview_id = res.json()["interview_id"]

    # Start Interview
    res = client.post(
        f"/interviews/{interview_id}/start",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code == 200

    questions = res.json()["questions"]
    assert len(questions) > 0

    # Submit answers (only 3 for testing)
    for q in questions[:3]:
        res = client.post(
            f"/interviews/{interview_id}/answers",
            json={
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "candidate_answer": "This is a test answer covering key concepts."
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert res.status_code == 200
        assert "final_score" in res.json()

    # Complete interview
    res = client.post(
        f"/interviews/{interview_id}/complete",
        headers={"Authorization": f"Bearer {token}"}
    )

    assert res.status_code == 200
    data = res.json()

    assert "overall_score" in data
    assert "selection_decision" in data


def test_recruiter_flow():
    # Register recruiter
    res = client.post("/recruiter/register", json={
        "full_name": "Recruiter",
        "email": "recruiter@test.com",
        "company": "TestCorp",
        "domain_focus": "Python",
        "password": "password123"
    })

    assert res.status_code == 201
    token = res.json()["access_token"]

    # Fetch candidates (may be empty initially)
    res = client.get(
        "/recruiter/candidates",
        headers={"Authorization": f"Bearer {token}"}
    )

    assert res.status_code == 200
    assert isinstance(res.json(), list)
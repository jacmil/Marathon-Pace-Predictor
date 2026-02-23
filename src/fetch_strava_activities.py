"""
Fetch all activities from the Strava API and write them to a JSONL file.

Requires a .env file with:
    STRAVA_CLIENT_ID
    STRAVA_CLIENT_SECRET
    STRAVA_REFRESH_TOKEN
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.environ["STRAVA_CLIENT_ID"]
CLIENT_SECRET = os.environ["STRAVA_CLIENT_SECRET"]
REFRESH_TOKEN = os.environ["STRAVA_REFRESH_TOKEN"]

TOKEN_URL = "https://www.strava.com/oauth/token"
ACTIVITIES_URL = "https://www.strava.com/api/v3/athlete/activities"
OUTPUT_FILE = "strava_data.jsonl"
PER_PAGE = 200  # max allowed by Strava


def get_access_token() -> str:
    """Exchange the refresh token for a fresh access token."""
    resp = requests.post(TOKEN_URL, data={
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "grant_type": "refresh_token",
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_all_activities(access_token: str) -> list[dict]:
    """Paginate through all activities."""
    headers = {"Authorization": f"Bearer {access_token}"}
    all_activities = []
    page = 1

    while True:
        print(f"Fetching page {page}...")
        resp = requests.get(
            ACTIVITIES_URL,
            headers=headers,
            params={"per_page": PER_PAGE, "page": page},
        )
        resp.raise_for_status()
        activities = resp.json()

        if not activities:
            break

        all_activities.extend(activities)
        page += 1

        # respect Strava's rate limits (100 requests per 15 min, 1000 per day)
        time.sleep(1)

    return all_activities


def main():
    print("Refreshing access token...")
    access_token = get_access_token()

    print("Fetching activities...")
    activities = fetch_all_activities(access_token)

    with open(OUTPUT_FILE, "w") as f:
        for activity in activities:
            f.write(json.dumps(activity) + "\n")

    print(f"Done. Wrote {len(activities)} activities to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

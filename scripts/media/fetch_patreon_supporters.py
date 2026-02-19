#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from platformdirs import user_config_dir
import sys

import aiohttp
import yaml


logger = logging.getLogger(__name__)

CONFIG_DIR = Path(user_config_dir("rayforge"))
PATREON_CONFIG_FILE = CONFIG_DIR / "patreon.yaml"
PATREON_API_BASE = "https://www.patreon.com/api/oauth2/v2"


def load_config():
    if PATREON_CONFIG_FILE.exists():
        with open(PATREON_CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    return {}


async def get_campaign_id(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{PATREON_API_BASE}/campaigns"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            if data.get("data"):
                return data["data"][0]["id"]
    return None


async def get_supporters(access_token, campaign_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    url = (
        f"{PATREON_API_BASE}/campaigns/{campaign_id}/members"
        f"?include=currently_entitled_tiers,user"
        f"&fields[member]=full_name,pledge_relationship_start,"
        f"last_charge_date,lifetime_support_cents,is_follower"
        f"&fields[user]=email"
        f"&sort=pledge_relationship_start"
    )
    supporters = []
    async with aiohttp.ClientSession() as session:
        while url:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
                supporters.extend(data.get("data", []))
                url = data.get("links", {}).get("next")
    return supporters


def is_paying_supporter(attrs):
    last_charge = attrs.get("last_charge_date")
    lifetime = attrs.get("lifetime_support_cents", 0)
    return last_charge is not None or lifetime > 0


async def main():
    parser = argparse.ArgumentParser(description="Fetch Patreon supporters")
    parser.add_argument(
        "--all",
        action="store_true",
        help="List all supporters, not just paying ones",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = load_config()
    access_token = config.get("access_token")

    if not access_token:
        print("No access token found in config file.")
        print(f"\nConfig file location: {PATREON_CONFIG_FILE}")
        print("\nTo add an access token:")
        print(
            "1. Go to: https://www.patreon.com/portal/registration/"
            "register-creator"
        )
        print("2. Select your app")
        print("3. Click 'Create a Creator's Access Token'")
        print("4. Add the token to the config file as 'access_token'")
        print("\nExample config file content:")
        print("  access_token: YOUR_TOKEN_HERE")
        sys.exit(1)

    campaign_id = await get_campaign_id(access_token)
    if not campaign_id:
        print("No campaign found for this account.")
        sys.exit(1)

    print(f"\nFetching supporters for campaign: {campaign_id}")
    supporters = await get_supporters(access_token, campaign_id)

    if not args.all:
        supporters = [
            s
            for s in supporters
            if is_paying_supporter(s.get("attributes", {}))
        ]

    print(f"\nFound {len(supporters)} supporters:\n")
    for supporter in supporters:
        attrs = supporter.get("attributes", {})
        name = attrs.get("full_name", "N/A")
        pledge_start = attrs.get("pledge_relationship_start", "N/A")
        last_charge = attrs.get("last_charge_date", "N/A")
        lifetime = attrs.get("lifetime_support_cents", 0) / 100
        is_follower = attrs.get("is_follower", False)

        included = supporter.get("included", [])
        email = "N/A"
        tier = "N/A"
        for item in included:
            if item.get("type") == "user":
                email = item.get("attributes", {}).get("email", "N/A")
            elif item.get("type") == "tier":
                tier = item.get("attributes", {}).get("title", "N/A")

        print(f"  Name: {name}")
        print(f"  Email: {email}")
        print(f"  Tier: {tier}")
        print(f"  Pledge Start: {pledge_start}")
        print(f"  Last Charge: {last_charge}")
        print(f"  Lifetime Support: ${lifetime:.2f}")
        print(f"  Follower: {is_follower}")
        print()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

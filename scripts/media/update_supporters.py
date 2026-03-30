#!/usr/bin/env python3

import asyncio
import logging
import re
import sys
from pathlib import Path

import aiohttp
import yaml
from platformdirs import user_config_dir


logger = logging.getLogger(__name__)

CONFIG_DIR = Path(user_config_dir("rayforge"))
PATREON_CONFIG_FILE = CONFIG_DIR / "patreon.yaml"
PATREON_API_BASE = "https://www.patreon.com/api/oauth2/v2"
SUPPORTERS_FILE = Path(__file__).resolve().parent.parent.parent / (
    "media/supporters.md"
)

SECTION_NAMED = "## Agreed to be mentioned"
SECTION_ANONYMOUS = (
    '## Did **not** agree to be mentioned (should be mentioned as "anonymous")'
)


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


async def fetch_supporters(access_token, campaign_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    url = (
        f"{PATREON_API_BASE}/campaigns/{campaign_id}/members"
        f"?include=currently_entitled_tiers,user"
        f"&fields[member]=full_name,pledge_relationship_start,"
        f"last_charge_date,lifetime_support_cents"
        f"&fields[user]=email"
        f"&fields[tier]=title"
        f"&sort=pledge_relationship_start"
    )
    members = []
    included = []
    async with aiohttp.ClientSession() as session:
        while url:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
                members.extend(data.get("data", []))
                included.extend(data.get("included", []))
                url = data.get("links", {}).get("next")
    return members, included


def is_paying_supporter(attrs):
    last_charge = attrs.get("last_charge_date")
    lifetime = attrs.get("lifetime_support_cents", 0)
    return last_charge is not None or lifetime > 0


def resolve_tier(member, included_lookup):
    tier_id = None
    rels = member.get("relationships", {})
    tiers_rel = rels.get("currently_entitled_tiers", {})
    tier_data = tiers_rel.get("data", [])
    logger.debug(
        "Resolving tier for %s, tier_data=%s",
        member.get("attributes", {}).get("full_name"),
        tier_data,
    )
    if tier_data:
        tier_id = tier_data[0]["id"]
    if tier_id and tier_id in included_lookup:
        title = (
            included_lookup[tier_id]
            .get("attributes", {})
            .get("title", "Supporter")
        )
        logger.debug("Found tier: %s (id=%s)", title, tier_id)
        return title
    logger.warning(
        "Could not resolve tier for %s (tier_id=%s, lookup_keys=%s)",
        member.get("attributes", {}).get("full_name"),
        tier_id,
        [k for k in included_lookup if isinstance(k, str)],
    )
    return "Supporter"


def build_included_lookup(included):
    lookup = {}
    for item in included:
        lookup[(item.get("type"), item["id"])] = item
        lookup[item["id"]] = item
    return lookup


def format_entry(name, date_str, tier):
    date_part = date_str[:10] if date_str else "unknown"
    return f"{date_part} {name} ({tier})"


def parse_existing_names(content):
    names = set()
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^\d{4}-\d{2}-\d{2}\s+(.+?)(?:\s+\(.+\))?$", line)
        if m:
            names.add(m.group(1).strip().lower())
        else:
            names.add(line.strip().lower())
    return names


def parse_file_sections(content):
    lines = content.splitlines()
    sections = {"header": [], "named": [], "anonymous": [], "past": []}
    current = "header"
    for line in lines:
        if line.strip() == SECTION_NAMED:
            current = "named"
            sections[current].append(line)
            continue
        elif line.strip() == SECTION_ANONYMOUS:
            current = "anonymous"
            sections[current].append(line)
            continue
        elif line.strip() == "# Past Supporters":
            current = "past"
            sections[current].append(line)
            continue
        sections[current].append(line)
    return sections


async def main():
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

    print(f"Fetching supporters for campaign: {campaign_id}")
    members, included = await fetch_supporters(access_token, campaign_id)
    included_lookup = build_included_lookup(included)

    paying = [
        m for m in members if is_paying_supporter(m.get("attributes", {}))
    ]
    print(f"Found {len(paying)} paying supporters")

    if not SUPPORTERS_FILE.exists():
        print(f"Error: {SUPPORTERS_FILE} not found.")
        sys.exit(1)

    content = SUPPORTERS_FILE.read_text()
    existing_names = parse_existing_names(content)
    sections = parse_file_sections(content)

    new_entries = []
    for member in paying:
        attrs = member.get("attributes", {})
        name = attrs.get("full_name", "Unknown")
        pledge_start = attrs.get("pledge_relationship_start")
        tier = resolve_tier(member, included_lookup)

        if name.strip().lower() in existing_names:
            continue

        new_entries.append(format_entry(name, pledge_start, tier))
        existing_names.add(name.strip().lower())

    if not new_entries:
        print("No new supporters to add.")
        return

    new_entries.sort()
    for entry in new_entries:
        print(f"  Adding: {entry}")

    anonymous_lines = sections["anonymous"]
    insert_idx = len(anonymous_lines)
    for i, line in enumerate(anonymous_lines):
        if line.strip() == "" or line.strip().startswith("*None"):
            insert_idx = i
            break

    for entry in reversed(new_entries):
        anonymous_lines.insert(insert_idx, entry)

    result = (
        sections["header"]
        + sections["named"]
        + sections["anonymous"]
        + sections["past"]
    )
    SUPPORTERS_FILE.write_text("\n".join(result) + "\n")
    print(f"\nAdded {len(new_entries)} new supporter(s) to {SUPPORTERS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())

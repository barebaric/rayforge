#!/usr/bin/env python3
"""
Fetch download statistics from various sources.

Sources:
- GitHub Releases
- Snap Store
- Flathub
- PyPI

Output options:
- json: Print stats as JSON
- monthly: Print monthly breakdown
- file: Write stats to JSON file
- metrics: Push metrics to VictoriaMetrics via Prometheus remote write
"""

import argparse
import base64
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, UTC
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json


METRICS_URL = os.environ.get("METRICS_URL", "")
METRICS_API_USER = os.environ.get("METRICS_API_USER", "")
METRICS_API_PASSWORD = os.environ.get("METRICS_API_PASSWORD", "")


def fetch_json(url, headers=None):
    req = Request(url, headers=headers or {})
    req.add_header("Accept", "application/json")
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def get_github_releases(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    data = fetch_json(url)
    total = 0
    by_version = {}
    for release in data:
        version = release["tag_name"]
        count = sum(
            asset.get("download_count", 0)
            for asset in release.get("assets", [])
        )
        by_version[version] = count
        total += count
    return {"total": total, "by_version": by_version}


def get_snap_downloads(snap_name):
    url = f"https://api.snapcraft.io/v2/snaps/info/{snap_name}"
    headers = {"Snap-Device-Series": "16"}
    try:
        data = fetch_json(url, headers)
        channel_map = data.get("channel-map", [])
        by_channel = {}
        first_release = None
        for channel in channel_map:
            channel_info = channel.get("channel", {})
            channel_name = channel_info.get("name", "unknown")
            version = channel.get("version", "unknown")
            released_at = channel_info.get("released-at")
            by_channel[channel_name] = {
                "version": version,
                "released_at": released_at,
            }
            if released_at and (
                first_release is None or released_at < first_release
            ):
                first_release = released_at
        return {
            "total": 0,
            "by_channel": by_channel,
            "first_release": first_release,
            "note": "Snap download counts require snapcraft CLI",
        }
    except (HTTPError, URLError):
        return {"total": 0, "by_channel": {}, "error": "Snap not found"}


def get_snap_downloads_cli(snap_name, start_date=None):
    result = {"total": 0, "by_month": {}, "error": None}
    cred = os.environ.get("SNAPCRAFT_STORE_CREDENTIALS")
    env = os.environ.copy()
    if cred:
        env["SNAPCRAFT_STORE_CREDENTIALS"] = cred
    try:
        cmd = [
            "snapcraft",
            "metrics",
            snap_name,
            "--name",
            "daily_device_change",
            "--format=json",
        ]
        if start_date:
            cmd.extend(["--start", start_date])
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, env=env
        )
        if proc.returncode != 0:
            result["error"] = f"snapcraft CLI failed: {proc.stderr.strip()}"
            return result
        metric = json.loads(proc.stdout)
        for series in metric.get("series", []):
            if series.get("name") == "new":
                values = series.get("values", [])
                dates = metric.get("buckets", [])
                monthly = defaultdict(int)
                total = 0
                for date_str, val in zip(dates, values):
                    if val is not None:
                        month = date_str[:7]
                        count = int(val)
                        monthly[month] += count
                        total += count
                result["total"] = total
                result["by_month"] = dict(sorted(monthly.items()))
                break
    except FileNotFoundError:
        result["error"] = "snapcraft CLI not installed"
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        result["error"] = str(e)
    return result


def get_flathub_downloads(app_id):
    url = f"https://flathub.org/api/v2/stats/{app_id}"
    try:
        data = fetch_json(url)
        if data is None:
            return {
                "total": 0,
                "installs_last_month": 0,
                "error": "App not on Flathub",
            }
        return {
            "total": data.get("installs_total", 0),
            "installs_last_month": data.get("installs_last_month", 0),
            "installs_last_7_days": data.get("installs_last_7_days", 0),
        }
    except (HTTPError, URLError):
        return {
            "total": 0,
            "installs_last_month": 0,
            "error": "Flathub API error",
        }


def get_pypi_downloads(package_name):
    url = f"https://pypistats.org/api/packages/{package_name}/recent"
    try:
        data = fetch_json(url)
        return {
            "last_day": data.get("data", {}).get("last_day", 0),
            "last_week": data.get("data", {}).get("last_week", 0),
            "last_month": data.get("data", {}).get("last_month", 0),
        }
    except (HTTPError, URLError):
        return {"last_day": 0, "last_week": 0, "last_month": 0}


def get_flathub_monthly(app_id):
    url = f"https://flathub.org/api/v2/stats/{app_id}"
    try:
        data = fetch_json(url)
        if data is None:
            return {}
        daily = data.get("installs_per_day", {})
        monthly = defaultdict(int)
        for date_str, count in daily.items():
            month = date_str[:7]
            monthly[month] += count
        return dict(sorted(monthly.items()))
    except (HTTPError, URLError):
        return {}


def get_pypi_monthly(package_name):
    url = f"https://pypistats.org/api/packages/{package_name}/overall?mirrors=false"
    try:
        data = fetch_json(url)
        daily = data.get("data", [])
        monthly = defaultdict(int)
        for entry in daily:
            date_str = entry.get("date", "")
            if date_str:
                month = date_str[:7]
                monthly[month] += entry.get("downloads", 0)
        return dict(sorted(monthly.items()))
    except (HTTPError, URLError):
        return {}


def send_to_victoriametrics(metrics):
    lines = []
    for metric in metrics:
        name = metric["name"]
        value = metric["value"]
        tags = metric.get("tags", {})

        tags_str = ",".join(f'{k}="{v}"' for k, v in tags.items())
        if tags_str:
            tags_str = "{" + tags_str + "}"

        lines.append(f"{name}{tags_str} {value}")

    url = f"{METRICS_URL}/api/v1/import/prometheus"
    data = "\n".join(lines).encode()

    credentials = base64.b64encode(
        f"{METRICS_API_USER}:{METRICS_API_PASSWORD}".encode()
    ).decode()

    req = Request(url, data=data, method="POST")
    req.add_header("Content-Type", "text/plain")
    req.add_header("Authorization", f"Basic {credentials}")

    try:
        with urlopen(req, timeout=30) as resp:
            return resp.status == 204
    except HTTPError as e:
        print(f"Failed to send to VictoriaMetrics: {e}")
        print(f"Response: {e.read().decode()}")
        return False
    except URLError as e:
        print(f"Failed to send to VictoriaMetrics: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch download stats")
    parser.add_argument("--github-owner", default="barebaric")
    parser.add_argument("--github-repo", default="rayforge")
    parser.add_argument("--snap-name", default="rayforge")
    parser.add_argument("--flathub-id", default="org.rayforge.rayforge")
    parser.add_argument("--pypi-package", default="rayforge")
    parser.add_argument(
        "--output",
        choices=["json", "monthly", "file", "metrics"],
        default="json",
    )
    parser.add_argument(
        "--output-file",
        default="website/static/stats.json",
        help="Output file for stats (default: website/static/stats.json)",
    )
    args = parser.parse_args()

    if args.output == "monthly":
        print("Fetching monthly download stats...")
        flathub_monthly = get_flathub_monthly(args.flathub_id)
        pypi_monthly = get_pypi_monthly(args.pypi_package)
        snap_data = get_snap_downloads_cli(args.snap_name, "2024-01-01")
        snap_monthly = snap_data.get("by_month", {})
        snap_error = snap_data.get("error")

        all_months = sorted(
            set(flathub_monthly.keys())
            | set(pypi_monthly.keys())
            | set(snap_monthly.keys())
        )

        print("\nMonthly Downloads by Source:")
        print("-" * 60)
        print(
            f"{'Month':<10} {'Flathub':>10} {'PyPI':>10} "
            f"{'Snap':>10} {'Total':>10}"
        )
        print("-" * 60)

        for month in all_months:
            fh = flathub_monthly.get(month, 0)
            pypi = pypi_monthly.get(month, 0)
            snap = snap_monthly.get(month, 0)
            total = fh + pypi + snap
            print(
                f"{month:<10} {fh:>10,} {pypi:>10,} {snap:>10,} {total:>10,}"
            )

        print("-" * 60)
        fh_total = sum(flathub_monthly.values())
        pypi_total = sum(pypi_monthly.values())
        snap_total = sum(snap_monthly.values())
        print(
            f"{'TOTAL':<10} "
            f"{fh_total:>10,} "
            f"{pypi_total:>10,} "
            f"{snap_total:>10,} "
            f"{fh_total + pypi_total + snap_total:>10,}"
        )
        if snap_error:
            print(f"\nNote: Snap stats unavailable - {snap_error}")
        return 0

    stats = {
        "timestamp": datetime.now(UTC).isoformat(),
        "github": get_github_releases(args.github_owner, args.github_repo),
        "snap": get_snap_downloads(args.snap_name),
        "flathub": get_flathub_downloads(args.flathub_id),
        "pypi": get_pypi_downloads(args.pypi_package),
    }

    stats["total_downloads"] = (
        stats["github"]["total"]
        + stats["snap"]["total"]
        + stats["flathub"]["total"]
        + stats["pypi"]["last_month"]
    )

    if args.output == "file":
        flathub_monthly = get_flathub_monthly(args.flathub_id)
        pypi_monthly = get_pypi_monthly(args.pypi_package)
        snap_data = get_snap_downloads_cli(args.snap_name, "2024-01-01")
        snap_monthly = snap_data.get("by_month", {})

        all_months = sorted(
            set(flathub_monthly.keys())
            | set(pypi_monthly.keys())
            | set(snap_monthly.keys())
        )

        monthly = []
        for month in all_months:
            fh = flathub_monthly.get(month, 0)
            pypi = pypi_monthly.get(month, 0)
            snap = snap_monthly.get(month, 0)
            monthly.append(
                {
                    "month": month,
                    "flathub": fh,
                    "pypi": pypi,
                    "snap": snap,
                    "total": fh + pypi + snap,
                }
            )

        full_stats = {
            "timestamp": datetime.now(UTC).isoformat(),
            "totals": {
                "github": stats["github"]["total"],
                "flathub": stats["flathub"]["total"],
                "pypi": stats["pypi"]["last_month"],
                "snap": snap_data.get("total", 0),
            },
            "github_by_version": stats["github"]["by_version"],
            "monthly": monthly,
        }

        with open(args.output_file, "w") as f:
            json.dump(full_stats, f, indent=2)
        print(f"Stats written to {args.output_file}")
        return 0

    if args.output == "metrics":
        if not METRICS_URL or not METRICS_API_USER or not METRICS_API_PASSWORD:
            print(
                "Error: METRICS_URL, METRICS_API_USER, and METRICS_API_PASSWORD must be set"
            )
            return 1

        metrics = [
            {
                "name": "downloads_total",
                "value": stats["github"]["total"],
                "tags": {"source": "github"},
            },
            {
                "name": "downloads_total",
                "value": stats["flathub"]["total"],
                "tags": {"source": "flathub"},
            },
            {
                "name": "downloads_total",
                "value": stats["pypi"]["last_month"],
                "tags": {"source": "pypi"},
            },
            {
                "name": "downloads_recent",
                "value": stats["flathub"].get("installs_last_7_days", 0),
                "tags": {"source": "flathub", "period": "7d"},
            },
            {
                "name": "downloads_recent",
                "value": stats["flathub"].get("installs_last_month", 0),
                "tags": {"source": "flathub", "period": "30d"},
            },
            {
                "name": "downloads_recent",
                "value": stats["pypi"]["last_day"],
                "tags": {"source": "pypi", "period": "1d"},
            },
            {
                "name": "downloads_recent",
                "value": stats["pypi"]["last_week"],
                "tags": {"source": "pypi", "period": "7d"},
            },
            {
                "name": "downloads_recent",
                "value": stats["pypi"]["last_month"],
                "tags": {"source": "pypi", "period": "30d"},
            },
        ]

        for version, count in stats["github"].get("by_version", {}).items():
            metrics.append(
                {
                    "name": "downloads_by_version",
                    "value": count,
                    "tags": {"source": "github", "version": version},
                }
            )

        if send_to_victoriametrics(metrics):
            print(
                f"Successfully sent {len(metrics)} metrics to VictoriaMetrics"
            )
            return 0
        else:
            print("Failed to send metrics to VictoriaMetrics")
            return 1

    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

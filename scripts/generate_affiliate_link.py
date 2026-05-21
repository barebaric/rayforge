import argparse
import configparser
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Any

API_DOMAIN = "api-sg.aliexpress.com"
API_PATH = "/sync"


def load_config() -> dict:
    path = os.path.expanduser("~/.config/aliexpress.key")
    if not os.path.exists(path):
        print(f"error: {path} not found", file=sys.stderr)
        print("create it with:", file=sys.stderr)
        print("  [aliexpress]", file=sys.stderr)
        print("  app_key=533956", file=sys.stderr)
        print("  app_secret=...", file=sys.stderr)
        print("  tracking_id=default", file=sys.stderr)
        print("  app_signature=rayforge", file=sys.stderr)
        sys.exit(1)
    cfg = configparser.ConfigParser()
    cfg.read(path)
    if "aliexpress" not in cfg:
        print(
            f"error: [aliexpress] section missing in {path}", file=sys.stderr
        )
        sys.exit(1)
    return {
        "app_key": cfg["aliexpress"]["app_key"],
        "app_secret": cfg["aliexpress"]["app_secret"],
        "tracking_id": cfg["aliexpress"].get("tracking_id", "default"),
        "app_signature": cfg["aliexpress"].get("app_signature", "rayforge"),
    }


def sign(secret: str, params: dict) -> str:
    keys = sorted(params.keys())
    s = "".join(f"{k}{params[k]}" for k in keys)
    s = f"{secret}{s}{secret}"
    return hashlib.md5(s.encode("utf-8")).hexdigest().upper()


def api_call(method: str, api_params: dict, cfg: dict) -> dict:
    timestamp = str(int(time.time() * 1000))
    sys_params = {
        "method": method,
        "app_key": cfg["app_key"],
        "sign_method": "md5",
        "timestamp": timestamp,
        "format": "json",
        "v": "2.0",
    }
    sign_params = {**sys_params, **api_params}
    sys_params["sign"] = sign(cfg["app_secret"], sign_params)
    qs = urllib.parse.urlencode(sorted(sys_params.items()))
    url = f"http://{API_DOMAIN}{API_PATH}?{qs}"
    body = urllib.parse.urlencode(api_params).encode("utf-8")
    req = urllib.request.Request(url, data=body)
    req.add_header(
        "Content-Type", "application/x-www-form-urlencoded;charset=utf-8"
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def generate_link(product_url: str, cfg: dict) -> str:
    result = api_call(
        "aliexpress.affiliate.link.generate",
        {
            "app_signature": cfg["app_signature"],
            "promotion_link_type": "0",
            "source_values": product_url,
            "tracking_id": cfg["tracking_id"],
        },
        cfg,
    )
    resp = result.get("aliexpress_affiliate_link_generate_response", {})
    resp_result = resp.get("resp_result", {})
    results = resp_result.get("result", {})
    links = results.get("promotion_links", {}).get("promotion_link", [])
    if not links:
        print("error: no affiliate link in response", file=sys.stderr)
        print(json.dumps(result, indent=2), file=sys.stderr)
        sys.exit(1)
    if "promotion_link" not in links[0]:
        msg = links[0].get("message", "unknown error")
        print(f"error: {msg}", file=sys.stderr)
        sys.exit(1)
    return links[0]["promotion_link"]


def product_query(keyword: str, cfg: dict) -> dict:
    result = api_call(
        "aliexpress.affiliate.product.query",
        {
            "keywords": keyword,
            "page_no": "1",
            "page_size": "5",
        },
        cfg,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate an Aliexpress affiliate link"
    )
    parser.add_argument("url", nargs="?", help="Aliexpress product URL")
    parser.add_argument("--query", "-q", help="Search for products by keyword")
    args: Any = parser.parse_args()

    cfg = load_config()

    if args.query:
        result = product_query(args.query, cfg)
        print(json.dumps(result, indent=2))
        return

    if args.url:
        link = generate_link(args.url, cfg)
        print(link)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

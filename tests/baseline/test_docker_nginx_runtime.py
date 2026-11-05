"""Runtime contracts for the Docker web reverse proxy."""

from pathlib import Path


def test_web_proxy_resolves_api_through_docker_dns_at_request_time():
    config = (Path(__file__).parents[2] / "docker" / "nginx.conf").read_text(encoding="utf-8")

    assert "resolver 127.0.0.11" in config
    assert "set $api_upstream http://api:8787;" in config
    assert "proxy_pass $api_upstream;" in config
    assert "proxy_pass http://api:8787;" not in config

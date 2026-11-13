import json

import pytest

from mistralai.workflows.testing.test_worker import create_test_worker
from worker.workflows import repo_digest as rd

MERGED_RAW = {
    "number": 101,
    "title": "Add retry logic to webhook delivery",
    "body": "Retries failed webhook deliveries with exponential backoff.",
    "html_url": "https://github.com/octo/hello/pull/101",
    "user": {"login": "ada"},
    "draft": False,
    "pull_request": {"merged_at": "2026-02-01T10:00:00Z"},
}

OPEN_RAW = {
    "number": 102,
    "title": "Fix flaky test in CI",
    "body": "Marks the timing-sensitive test as flaky and retries it once.",
    "html_url": "https://github.com/octo/hello/pull/102",
    "user": {"login": "grace"},
    "draft": False,
    "pull_request": {},
}


def test_parse_repo_handle_accepts_handle_url_and_git_suffix():
    assert rd.parse_repo_handle("octo/hello") == ("octo", "hello", "octo/hello")
    assert rd.parse_repo_handle("https://github.com/octo/hello") == ("octo", "hello", "octo/hello")
    assert rd.parse_repo_handle("https://github.com/octo/hello.git") == ("octo", "hello", "octo/hello")


def test_parse_repo_handle_rejects_garbage():
    with pytest.raises(ValueError):
        rd.parse_repo_handle("not-a-repo")
    with pytest.raises(ValueError):
        rd.parse_repo_handle("https://example.com/octo/hello")


def test_validate_since():
    assert rd.validate_since(" 2026-01-31 ") == "2026-01-31"
    with pytest.raises(ValueError):
        rd.validate_since("01/31/2026")


def test_build_search_query():
    assert (
        rd.build_search_query("octo/hello", merged=True, since="2026-01-01")
        == "repo:octo/hello is:pr is:merged merged:>=2026-01-01"
    )
    assert (
        rd.build_search_query("octo/hello", merged=False)
        == "repo:octo/hello is:pr is:open -is:draft"
    )


def test_coerce_search_items_handles_items_content_and_plain_list():
    assert rd._coerce_search_items({"items": [{"number": 1}]}) == [{"number": 1}]
    mcp_style = {
        "content": [
            {"type": "text", "text": json.dumps({"items": [{"number": 2}]})},
        ]
    }
    assert rd._coerce_search_items(mcp_style) == [{"number": 2}]
    assert rd._coerce_search_items([{"number": 3}]) == [{"number": 3}]
    assert rd._coerce_search_items(None) == []
    assert rd._coerce_search_items({"unexpected": 1}) == []


def test_normalize_pr_truncates_body_and_reads_author():
    pr = rd.normalize_pr({**MERGED_RAW, "body": "x" * 5000})
    assert pr.number == 101
    assert pr.author == "ada"
    assert pr.merged_at == "2026-02-01T10:00:00Z"
    assert len(pr.body) == rd.BODY_SNIPPET_CHARS


def test_format_digest_renders_both_sections_with_links():
    merged = [rd.DigestItem(number=101, title="Add retry logic", url="https://x/101", summary="Adds backoff retries <safe>")]
    awaiting = [rd.DigestItem(number=102, title="Fix flaky test", url="https://x/102", summary="Retries flaky test")]
    digest = rd.format_digest("octo/hello", "2026-01-01", merged, awaiting)
    assert "*Repo digest: octo/hello*" in digest
    assert "<https://x/101|#101 Add retry logic>" in digest
    assert "Adds backoff retries &lt;safe&gt;" in digest
    assert "*Awaiting review (1)*" in digest
    assert "<https://x/102|#102 Fix flaky test>" in digest


def test_format_digest_handles_empty_sections():
    digest = rd.format_digest("octo/hello", "2026-01-01", [], [])
    assert digest.count("_None_") == 2


def test_digest_items_fall_back_to_title_when_summary_missing():
    items = rd._digest_items([rd.normalize_pr(OPEN_RAW)], {})
    assert items[0].summary == OPEN_RAW["title"]
    items = rd._digest_items([rd.normalize_pr(OPEN_RAW)], {102: "one-liner"})
    assert items[0].summary == "one-liner"


async def test_search_pull_requests_paginates_and_stops(temporal_env_with_converter, monkeypatch):
    calls = []

    async def fake_call_tool(name, arguments):
        calls.append((name, dict(arguments)))
        if arguments["page"] == 1:
            return {"items": [{"number": i} for i in range(rd.PER_PAGE)]}
        return {"items": [{"number": 999}]}

    items = await rd._search_pull_requests(
        fake_call_tool, query="repo:o/r is:pr", owner="o", repo="r", sort="updated", order="desc"
    )
    assert len(items) == rd.PER_PAGE + 1
    assert calls[0][0] == "search_pull_requests"
    assert calls[0][1]["query"] == "repo:o/r is:pr"
    assert calls[1][1]["page"] == 2


async def test_post_digest_to_self_dm_flow(temporal_env_with_converter):
    sent = []

    async def fake_call_tool(name, arguments):
        if name == "slack_read_user_profile":
            return {"user": {"id": "UDV4TPUAE", "name": "me"}}
        if name == "slack_send_message":
            sent.append((name, dict(arguments)))
            return {"ok": True, "ts": "1700000000.000100", "channel": arguments["channel_id"]}
        raise AssertionError(f"unexpected tool {name}")

    post = await rd._post_digest_to_self_dm(fake_call_tool, "*digest*")
    assert post.channel == "UDV4TPUAE"
    assert post.ts == "1700000000.000100"
    assert sent == [("slack_send_message", {"channel_id": "UDV4TPUAE", "message": "*digest*"})]


async def test_post_digest_falls_back_to_create_conversation(temporal_env_with_converter):
    sent = []

    async def fake_call_tool(name, arguments):
        if name == "slack_read_user_profile":
            return {"id": "UDV4TPUAE"}
        if name == "slack_create_conversation":
            assert arguments == {"user_ids": ["UDV4TPUAE"]}
            return {"channel": {"id": "DSELF123"}}
        if name == "slack_send_message":
            sent.append(dict(arguments))
            if arguments["channel_id"] == "UDV4TPUAE":
                raise RuntimeError("channel_not_found")
            return {"ok": True, "ts": "42.0"}
        raise AssertionError(f"unexpected tool {name}")

    post = await rd._post_digest_to_self_dm(fake_call_tool, "hi")
    assert post.channel == "DSELF123"
    assert post.ts == "42.0"
    assert sent[-1] == {"channel_id": "DSELF123", "message": "hi"}


def test_format_digest_truncates_to_fit_slack_limit():
    many = [
        rd.DigestItem(number=i, title="t" * 200, url=f"https://x/{i}", summary="s" * 140)
        for i in range(10)
    ]
    digest = rd.format_digest("octo/hello", "2026-01-01", many, many)
    assert len(digest) <= rd.MAX_DIGEST_CHARS


async def test_repo_digest_workflow_end_to_end(temporal_env_with_converter, monkeypatch):
    async def fake_search(call_tool, *, query, owner, repo, sort, order):
        assert owner == "octo" and repo == "hello"
        if "is:merged" in query:
            assert "merged:>=2026-02-01" in query
            return [MERGED_RAW]
        assert "is:open" in query
        return [OPEN_RAW]

    async def fake_summarize(prs, section):
        return {pr.number: f"summary of {pr.number}" for pr in prs}

    async def fake_post(call_tool, digest_markdown):
        assert "#101" in digest_markdown and "#102" in digest_markdown
        return rd.SlackPost(channel="DSELF", ts="1.2")

    monkeypatch.setattr(rd, "_search_pull_requests", fake_search)
    monkeypatch.setattr(rd, "_summarize_prs", fake_summarize)
    monkeypatch.setattr(rd, "_post_digest_to_self_dm", fake_post)

    async with create_test_worker(
        temporal_env_with_converter,
        workflows=[rd.RepoDigestWorkflow],
        activities=[
            rd.fetch_merged_prs,
            rd.fetch_open_prs,
            rd.summarize_section,
            rd.post_digest_to_slack_self_dm,
        ],
    ):
        result = await temporal_env_with_converter.client.execute_workflow(
            rd.RepoDigestWorkflow,
            rd.RepoDigestInput(repo="octo/hello", since="2026-02-01"),
            id="test-repo-digest",
            task_queue="test-task-queue",
        )

    if isinstance(result, dict):
        result = rd.RepoDigestResult.model_validate(result)
    assert result.repo == "octo/hello"
    assert result.since == "2026-02-01"
    assert [item.number for item in result.merged] == [101]
    assert result.merged[0].summary == "summary of 101"
    assert result.merged[0].url == MERGED_RAW["html_url"]
    assert [item.number for item in result.awaiting_review] == [102]
    assert result.slack.channel == "DSELF"
    assert "*Merged since 2026-02-01 (1)*" in result.digest_markdown
    assert "*Awaiting review (1)*" in result.digest_markdown
    assert "<https://github.com/octo/hello/pull/101|#101 Add retry logic to webhook delivery>" in result.digest_markdown

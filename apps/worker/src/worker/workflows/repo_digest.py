"""Repo catch-up digest workflow.

Input: a GitHub repo handle ("owner/repo") and a since date ("YYYY-MM-DD").
The workflow lists pull requests merged since that date and pull requests
still open awaiting review, writes a one-line summary for each, and posts
the digest to the triggering user's Slack self-DM (the conversation with
themselves).

Runs on-behalf-of the triggering user: the GitHub App and Slack connectors
resolve that user's credentials, so the deployment must be hardened.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from typing import Any, Awaitable, Callable

import pydantic

import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.client import get_mistral_client
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

github_connector = connector("github_app")
slack_connector = connector("slack")

SUMMARY_MODEL = "mistral-small-latest"
MAX_ITEMS_PER_SECTION = 10
MAX_DIGEST_CHARS = 4800
PER_PAGE = 100
MAX_PAGES = 3
BODY_SNIPPET_CHARS = 800

GITHUB_SEARCH_FIELDS = [
    "number",
    "title",
    "body",
    "html_url",
    "user",
    "draft",
    "pull_request",
    "created_at",
    "updated_at",
]


class RepoDigestInput(pydantic.BaseModel):
    repo: str
    since: str


class PrInfo(pydantic.BaseModel):
    number: int
    title: str = ""
    url: str = ""
    body: str = ""
    author: str | None = None
    draft: bool = False
    merged_at: str | None = None


class DigestItem(pydantic.BaseModel):
    number: int
    title: str
    url: str
    summary: str
    author: str | None = None


class SlackPost(pydantic.BaseModel):
    channel: str | None = None
    ts: str | None = None


class RepoDigestResult(pydantic.BaseModel):
    repo: str
    since: str
    merged: list[DigestItem]
    awaiting_review: list[DigestItem]
    digest_markdown: str
    slack: SlackPost


_REPO_RE = re.compile(
    r"^(?:https?://github\.com/)?(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_SINCE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SLACK_USER_ID_RE = re.compile(r"U[A-Z0-9]{8,}")


def parse_repo_handle(handle: str) -> tuple[str, str, str]:
    match = _REPO_RE.match(handle.strip())
    if not match:
        raise ValueError(f"repo must be 'owner/repo' or a github.com URL, got {handle!r}")
    owner, repo = match.group("owner"), match.group("repo")
    return owner, repo, f"{owner}/{repo}"


def validate_since(since: str) -> str:
    since = since.strip()
    if not _SINCE_RE.match(since):
        raise ValueError(f"since must be a date in YYYY-MM-DD form, got {since!r}")
    return since


def build_search_query(repo_handle: str, *, merged: bool, since: str | None = None) -> str:
    query = f"repo:{repo_handle} is:pr"
    if merged:
        query += f" is:merged merged:>={since}"
    else:
        query += " is:open -is:draft"
    return query


def _coerce_search_items(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("items", "results", "pull_requests", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        structured = payload.get("structuredContent") or payload.get("structured_content")
        if structured is not None:
            return _coerce_search_items(structured)
        content = payload.get("content")
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ):
                    try:
                        return _coerce_search_items(json.loads(block["text"]))
                    except (ValueError, TypeError):
                        continue
    return []


def normalize_pr(raw: dict[str, Any]) -> PrInfo:
    user = raw.get("user")
    pr = raw.get("pull_request")
    merged_at = pr.get("merged_at") if isinstance(pr, dict) else None
    return PrInfo(
        number=int(raw.get("number") or 0),
        title=str(raw.get("title") or ""),
        url=str(raw.get("html_url") or ""),
        body=str(raw.get("body") or "")[:BODY_SNIPPET_CHARS],
        author=user.get("login") if isinstance(user, dict) else None,
        draft=bool(raw.get("draft")),
        merged_at=merged_at,
    )


async def _search_pull_requests(
    call_tool: Callable[..., Awaitable[Any]],
    *,
    query: str,
    owner: str,
    repo: str,
    sort: str,
    order: str,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        response = await call_tool(
            "search_pull_requests",
            {
                "query": query,
                "owner": owner,
                "repo": repo,
                "fields": GITHUB_SEARCH_FIELDS,
                "perPage": PER_PAGE,
                "page": page,
                "sort": sort,
                "order": order,
            },
        )
        batch = _coerce_search_items(response)
        collected.extend(batch)
        if len(batch) < PER_PAGE:
            break
    return collected


def _summary_prompt(section: str, prs: list[PrInfo]) -> str:
    listing = json.dumps(
        [
            {
                "number": pr.number,
                "title": pr.title,
                "url": pr.url,
                "author": pr.author,
                "body": pr.body,
            }
            for pr in prs
        ],
        ensure_ascii=False,
    )
    return (
        f"Write one one-line summary for each pull request in the {section} section of a repo digest.\n"
        "Rules: exactly one line per PR, at most 140 characters, plain text with no markdown or links; "
        "describe what the PR changes based only on its title and body; do not invent details.\n"
        'Return ONLY a JSON object mapping the PR number as a string to its summary, e.g. {"123": "..."}.'
        f"\n\nPull requests:\n{listing}"
    )


async def _summarize_prs(prs: list[PrInfo], section: str) -> dict[int, str]:
    if not prs:
        return {}
    client = get_mistral_client()
    response = await client.chat.complete_async(
        model=SUMMARY_MODEL,
        messages=[{"role": "user", "content": _summary_prompt(section, prs)}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        content = "".join(
            chunk.get("text", "") for chunk in content if isinstance(chunk, dict)
        )
    try:
        parsed = json.loads(content or "")
    except (ValueError, TypeError):
        parsed = {}
    summaries: dict[int, str] = {}
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            try:
                summaries[int(key)] = str(value)
            except (ValueError, TypeError):
                continue
    return summaries


def _digest_items(prs: list[PrInfo], summaries: dict[int, str]) -> list[DigestItem]:
    items: list[DigestItem] = []
    for pr in prs[:MAX_ITEMS_PER_SECTION]:
        items.append(
            DigestItem(
                number=pr.number,
                title=pr.title,
                url=pr.url,
                summary=summaries.get(pr.number) or pr.title,
                author=pr.author,
            )
        )
    return items


def _slack_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_digest(
    repo_handle: str,
    since: str,
    merged: list[DigestItem],
    awaiting_review: list[DigestItem],
) -> str:
    lines = [
        f"*Repo digest: {repo_handle}* — merged since {since} and awaiting review",
        "",
        f"*Merged since {since} ({len(merged)})*",
    ]
    if merged:
        for item in merged:
            lines.append(
                f"• <{item.url}|#{item.number} {item.title}> — {_slack_escape(item.summary)}"
            )
    else:
        lines.append("_None_")
    lines.append("")
    lines.append(f"*Awaiting review ({len(awaiting_review)})*")
    if awaiting_review:
        for item in awaiting_review:
            lines.append(
                f"• <{item.url}|#{item.number} {item.title}> — {_slack_escape(item.summary)}"
            )
    else:
        lines.append("_None_")
    return "\n".join(lines)


def format_digest(
    repo_handle: str,
    since: str,
    merged: list[DigestItem],
    awaiting_review: list[DigestItem],
) -> str:
    merged = list(merged)
    awaiting_review = list(awaiting_review)
    while len(_render_digest(repo_handle, since, merged, awaiting_review)) > MAX_DIGEST_CHARS:
        longer = merged if len(merged) >= len(awaiting_review) else awaiting_review
        if not longer:
            break
        longer.pop()
    return _render_digest(repo_handle, since, merged, awaiting_review)


def _tool_list(payload: Any) -> list[dict[str, Any]]:
    tools = getattr(payload, "tools", None) if not isinstance(payload, dict) else payload.get("tools")
    if not isinstance(tools, list):
        return []
    result: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            result.append(tool)
        elif hasattr(tool, "model_dump"):
            result.append(tool.model_dump())
    return result


def _pick_tool(
    tools: list[dict[str, Any]],
    *,
    name_hints: tuple[str, ...],
    description_hints: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    for tool in tools:
        name = str(tool.get("name") or "").lower()
        if all(hint in name for hint in name_hints) and not any(ex in name for ex in exclude):
            return tool
    if description_hints:
        for tool in tools:
            name = str(tool.get("name") or "").lower()
            description = str(tool.get("description") or "").lower()
            if (
                all(hint in description for hint in description_hints)
                and not any(ex in name for ex in exclude)
            ):
                return tool
    return None


def _schema_properties(tool: dict[str, Any]) -> dict[str, Any]:
    schema = tool.get("input_schema") or tool.get("inputSchema") or {}
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _pick_arg(properties: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    lowered = {str(key).lower(): key for key in properties}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for key in properties:
        key_lower = str(key).lower()
        if any(candidate in key_lower for candidate in candidates):
            return key
    return None


def _dig_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in ("structuredContent", "structured_content", "result", "data"):
            if key in payload and payload[key] not in (None, {}, []):
                return _dig_json(payload[key])
        content = payload.get("content")
        if isinstance(content, list):
            texts = [
                block.get("text")
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ]
            for text in texts:
                try:
                    return json.loads(text)
                except (ValueError, TypeError):
                    continue
            if texts:
                return texts[0]
        return payload
    return payload


async def _post_digest_to_self_dm(
    call_tool: Callable[..., Awaitable[Any]],
    client_factory: Callable[..., Any],
    digest_markdown: str,
) -> SlackPost:
    client = client_factory(use_executor_credentials=True)
    tools_payload = await client.beta.connectors.list_tools_async(
        connector_id_or_name="slack", page_size=100
    )
    tools = _tool_list(tools_payload)
    if not tools:
        raise RuntimeError(
            "The Slack connector exposes no tools: connect Slack (OAuth) for the "
            "triggering user in Studio, or complete the auth URL emitted on first run."
        )

    send_tool = _pick_tool(
        tools,
        name_hints=("send", "message"),
        description_hints=("send", "message"),
        exclude=("draft", "schedule"),
    )
    if send_tool is None:
        raise RuntimeError("No send-message tool found on the Slack connector")

    profile_tool = (
        _pick_tool(tools, name_hints=("whoami",))
        or _pick_tool(tools, name_hints=("read", "user", "profile"))
        or _pick_tool(tools, name_hints=("user", "profile"))
        or _pick_tool(tools, name_hints=("user", "info"))
    )
    my_user_id: str | None = None
    if profile_tool is not None:
        profile_response = _dig_json(await call_tool(profile_tool.get("name"), {}))
        match = _SLACK_USER_ID_RE.search(json.dumps(profile_response))
        if match:
            my_user_id = match.group(0)
    if my_user_id is None:
        raise RuntimeError("Could not determine the triggering user's Slack user ID")

    send_props = _schema_properties(send_tool)
    channel_arg = _pick_arg(send_props, ("channel_id", "channel", "conversation"))
    text_arg = _pick_arg(send_props, ("message", "text"))
    if channel_arg is None or text_arg is None:
        raise RuntimeError(
            f"Send-message tool {send_tool.get('name')!r} lacks a channel/message argument: {list(send_props)}"
        )

    # slack_send_message accepts a user_id as channel_id, so the self-DM is the
    # triggering user's own ID; fall back to opening the DM conversation explicitly.
    channel_id = my_user_id
    try:
        send_response = _dig_json(
            await call_tool(send_tool.get("name"), {channel_arg: channel_id, text_arg: digest_markdown})
        )
    except Exception:
        create_tool = _pick_tool(tools, name_hints=("create", "conversation"))
        if create_tool is None:
            raise
        create_props = _schema_properties(create_tool)
        users_arg = _pick_arg(create_props, ("user_ids", "users", "user"))
        if users_arg is None:
            raise
        schema = create_tool.get("input_schema") or create_tool.get("inputSchema") or {}
        prop = (schema.get("properties") or {}).get(users_arg) or {}
        value: Any = [my_user_id] if prop.get("type") == "array" else my_user_id
        create_response = _dig_json(await call_tool(create_tool.get("name"), {users_arg: value}))
        channel = create_response.get("channel") if isinstance(create_response, dict) else None
        channel_id = (
            (channel or {}).get("id")
            or (create_response or {}).get("channel_id")
            or (create_response or {}).get("id")
        )
        if not channel_id:
            raise
        send_response = _dig_json(
            await call_tool(send_tool.get("name"), {channel_arg: channel_id, text_arg: digest_markdown})
        )

    ts = None
    if isinstance(send_response, dict):
        ts = send_response.get("ts") or send_response.get("message_ts")
        message = send_response.get("message")
        if ts is None and isinstance(message, dict):
            ts = message.get("ts")
    return SlackPost(channel=channel_id, ts=ts)


@workflows.activity(name="fetch-merged-prs", start_to_close_timeout=timedelta(minutes=5))
async def fetch_merged_prs(
    repo_handle: str,
    since: str,
    github: ToolCallClient = Depends(github_connector),
) -> list[PrInfo]:
    owner, repo, handle = parse_repo_handle(repo_handle)
    since = validate_since(since)
    raw = await _search_pull_requests(
        github.call_tool,
        query=build_search_query(handle, merged=True, since=since),
        owner=owner,
        repo=repo,
        sort="updated",
        order="desc",
    )
    return [normalize_pr(item) for item in raw]


@workflows.activity(name="fetch-open-prs", start_to_close_timeout=timedelta(minutes=5))
async def fetch_open_prs(
    repo_handle: str,
    github: ToolCallClient = Depends(github_connector),
) -> list[PrInfo]:
    owner, repo, handle = parse_repo_handle(repo_handle)
    raw = await _search_pull_requests(
        github.call_tool,
        query=build_search_query(handle, merged=False),
        owner=owner,
        repo=repo,
        sort="created",
        order="asc",
    )
    return [normalize_pr(item) for item in raw]


@workflows.activity(name="summarize-section", start_to_close_timeout=timedelta(minutes=5))
async def summarize_section(section: str, prs: list[PrInfo]) -> list[DigestItem]:
    summaries = await _summarize_prs(prs, section)
    return _digest_items(prs, summaries)


@workflows.activity(name="post-digest-to-slack-self-dm", start_to_close_timeout=timedelta(minutes=10))
async def post_digest_to_slack_self_dm(
    digest_markdown: str,
    slack: ToolCallClient = Depends(slack_connector),
) -> SlackPost:
    return await _post_digest_to_self_dm(slack.call_tool, get_mistral_client, digest_markdown)


@workflows.workflow.define(
    name="repo-digest",
    workflow_display_name="Repo catch-up digest",
    workflow_description=(
        "List pull requests merged since a given date and pull requests still open "
        "awaiting review on a GitHub repo, summarize each in one line, and post the "
        "digest to your Slack self-DM."
    ),
    on_behalf_of=True,
)
@uses_connectors(github_connector, slack_connector)
class RepoDigestWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: RepoDigestInput) -> RepoDigestResult:
        _, _, handle = parse_repo_handle(input.repo)
        since = validate_since(input.since)
        merged_prs = await fetch_merged_prs(handle, since)
        open_prs = await fetch_open_prs(handle)
        merged = await summarize_section("merged", merged_prs)
        awaiting_review = await summarize_section("awaiting review", open_prs)
        digest = format_digest(handle, since, merged, awaiting_review)
        slack_post = await post_digest_to_slack_self_dm(digest)
        return RepoDigestResult(
            repo=handle,
            since=since,
            merged=merged,
            awaiting_review=awaiting_review,
            digest_markdown=digest,
            slack=slack_post,
        )

"""Tests for DeepSeek (OpenAI-compatible) provider."""

import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models.anthropic import MessagesRequest, Message, Tool
from providers.base import ProviderConfig
from providers.deepseek import DEEPSEEK_DEFAULT_BASE, DeepSeekProvider
from providers.exceptions import InvalidRequestError


@pytest.fixture
def deepseek_config():
    return ProviderConfig(
        api_key="test_deepseek_key",
        base_url=DEEPSEEK_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""

    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def deepseek_provider(deepseek_config):
    return DeepSeekProvider(deepseek_config)


def test_init(deepseek_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = DeepSeekProvider(deepseek_config)
        assert provider._api_key == "test_deepseek_key"
        assert provider._base_url == DEEPSEEK_DEFAULT_BASE
        mock_openai.assert_called_once()


def test_default_base_url_constant():
    assert DEEPSEEK_DEFAULT_BASE == "https://api.deepseek.com/v1"


def test_build_request_body_basic(deepseek_provider):
    """Basic request body conversion attaches system message from Claude request."""
    request = MessagesRequest(
        model="deepseek-chat",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        system="S",
    )
    body = deepseek_provider._build_request_body(request)

    assert body["model"] == "deepseek-chat"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "S"
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1]["content"] == "Hello"
    assert body["max_tokens"] == 100


def test_build_request_body_global_disable_blocks_reasoning_mapping():
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="test_deepseek_key",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        )
    )
    request = MessagesRequest(
        model="deepseek-reasoner",
        messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="thinking", reasoning_content="reason"),
        ],
    )
    body = provider._build_request_body(request)

    for msg in body.get("messages", []):
        assert "reasoning_content" not in msg


def test_build_request_body_preserves_caller_extra_body(deepseek_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        extra_body={"metadata": {"user": "u1"}},
    )
    body = deepseek_provider._build_request_body(request)

    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("metadata") == {"user": "u1"}


def test_preflight_rejects_mcp_servers(deepseek_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        mcp_servers=[{"type": "url", "url": "https://x"}],
    )
    with pytest.raises(InvalidRequestError, match="mcp_servers"):
        deepseek_provider.preflight_stream(request)


def test_preflight_rejects_listed_server_tools_in_tools_list(deepseek_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        tools=[Tool(name="web_search", type="web_search_20250305", input_schema={})],
    )
    with pytest.raises(InvalidRequestError, match="web_search"):
        deepseek_provider.preflight_stream(request)


def test_preflight_rejects_server_tool_result_blocks(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "s1",
                            "name": "web_search",
                            "input": {"q": "a"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "s1",
                            "content": [],
                        },
                    ],
                }
            ],
        }
    )
    with pytest.raises(InvalidRequestError, match=r"web_search_tool_result|server"):
        deepseek_provider.preflight_stream(request)


def test_strips_image_block_inside_tool_result(deepseek_provider):
    """Image blocks nested inside tool_result.content are stripped, not rejected."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"path": "shot.png"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {"type": "text", "text": "screenshot saved"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abc",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    # In OpenAI format:
    # First user message is hello (or similar)
    # Assistant message with tool_calls
    # Tool response message
    tool_msg = body["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert "screenshot saved" in tool_msg["content"]
    assert "base64" not in tool_msg["content"]
    assert "abc" not in tool_msg["content"]


def test_image_only_tool_result_replaced_with_placeholder(deepseek_provider):
    """A tool_result whose only inner block is an image becomes a placeholder string."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Screenshot",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abc",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    tool_msg = body["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert "attachment omitted" in tool_msg["content"].lower()
    assert "image or document inputs" in tool_msg["content"].lower()


def test_document_only_tool_result_replaced_with_generic_placeholder(deepseek_provider):
    """A document-only tool_result uses the generic attachment placeholder."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "paper.pdf"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "file",
                                        "file_id": "file_pdf",
                                    },
                                },
                            ],
                        }
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    tool_msg = body["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert "attachment omitted" in tool_msg["content"].lower()
    assert "document inputs" in tool_msg["content"].lower()


def test_image_only_message_replaced_with_placeholder(deepseek_provider):
    """A top-level image-only message remains non-empty after stripping."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    user_msg = body["messages"][0]
    assert user_msg["role"] == "user"
    assert "attachment omitted" in user_msg["content"].lower()
    assert "image or document inputs" in user_msg["content"].lower()


def test_document_only_message_replaced_with_placeholder(deepseek_provider):
    """A top-level document-only message remains non-empty after stripping."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": "file_pdf"},
                        },
                    ],
                },
            ],
        }
    )

    body = deepseek_provider._build_request_body(request)

    user_msg = body["messages"][0]
    assert user_msg["role"] == "user"
    assert "attachment omitted" in user_msg["content"].lower()
    assert "document inputs" in user_msg["content"].lower()


def test_warns_when_stripping_attachment_blocks(deepseek_provider, caplog):
    """A warning is emitted when image/document blocks are dropped so users notice."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                },
            ],
        }
    )

    with caplog.at_level(logging.WARNING):
        deepseek_provider._build_request_body(request)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("stripped unsupported attachment blocks" in r.message for r in warnings)


def test_no_warning_when_no_attachments(deepseek_provider, caplog):
    """No warning is emitted on plain text-only requests."""
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    with caplog.at_level(logging.WARNING):
        deepseek_provider._build_request_body(request)

    assert not any(
        "stripped unsupported attachment blocks" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


@pytest.mark.asyncio
async def test_stream_response_text(deepseek_provider):
    """Text content deltas are emitted as text blocks."""
    req = MessagesRequest(
        model="deepseek-chat",
        messages=[Message(role="user", content="hello")],
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello back!",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        deepseek_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in deepseek_provider.stream_response(req)]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(deepseek_provider):
    """reasoning_content deltas are emitted as thinking blocks."""
    req = MessagesRequest(
        model="deepseek-reasoner",
        messages=[Message(role="user", content="hello")],
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking...",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        deepseek_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in deepseek_provider.stream_response(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )


@pytest.mark.asyncio
async def test_cleanup(deepseek_provider):
    deepseek_provider._client = AsyncMock()

    await deepseek_provider.cleanup()

    deepseek_provider._client.close.assert_called_once()

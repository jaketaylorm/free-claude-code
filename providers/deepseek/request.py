"""Request builder for DeepSeek (OpenAI-compatible chat completions)."""

from __future__ import annotations

import json
from copy import copy
from typing import Any

from loguru import logger

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError, get_block_attr, get_block_type
from providers.exceptions import InvalidRequestError

_UNSUPPORTED_MESSAGE_BLOCK_TYPES = frozenset(
    {
        "image",
        "document",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
    }
)

_OMITTED_ATTACHMENT_TEXT = (
    "[attachment omitted: DeepSeek does not support image or document inputs]"
)
_OMITTED_DOCUMENT_TEXT = (
    "[attachment omitted: DeepSeek does not support document inputs]"
)


def _is_server_listed_tool(tool: Any) -> bool:
    """True for Anthropic web_search / web_fetch-style tool definitions (listed tools)."""
    name = (getattr(tool, "name", "") or "").strip()
    if name in ("web_search", "web_fetch"):
        return True
    typ = getattr(tool, "type", None)
    if isinstance(typ, str):
        return typ.startswith("web_search") or typ.startswith("web_fetch")
    return False


def _walk_block_list_for_unsupported(blocks: Any, *, where: str) -> None:
    if not isinstance(blocks, list):
        return
    for block in blocks:
        btype = get_block_type(block)
        if btype in ("server_tool_use", "web_search_tool_result", "web_fetch_tool_result"):
            raise InvalidRequestError(
                f"DeepSeek native does not support {btype!r} blocks ({where})."
            )
        if btype == "tool_result":
            content = get_block_attr(block, "content")
            if isinstance(content, list):
                _walk_block_list_for_unsupported(
                    content, where=f"{where} (tool_result content)"
                )


def _validate_deepseek_request_data(request_data: Any) -> None:
    mcp = getattr(request_data, "mcp_servers", None)
    if mcp:
        raise InvalidRequestError(
            "DeepSeek native does not support mcp_servers on requests."
        )

    for tool in getattr(request_data, "tools", None) or ():
        if _is_server_listed_tool(tool):
            raise InvalidRequestError(
                "DeepSeek native does not support listed Anthropic server tools "
                "(web_search / web_fetch). Remove them or use a different provider."
            )

    for i, message in enumerate(getattr(request_data, "messages", None) or ()):
        c = message.content
        if isinstance(c, list):
            _walk_block_list_for_unsupported(c, where=f"messages[{i}].content")

    system = getattr(request_data, "system", None)
    if isinstance(system, list):
        _walk_block_list_for_unsupported(system, where="system")


def _clean_unsupported_blocks(messages: list[Any]) -> tuple[list[Any], bool]:
    """Clean unsupported attachment blocks from message contents to prevent conversion errors."""
    cleaned_messages = []
    any_stripped = False

    for msg in messages:
        # Shallow copy of message to avoid mutating the request in-place
        new_msg = copy(msg)
        content = new_msg.content

        if isinstance(content, list):
            new_content = []
            message_dropped_attachment = False
            for block in content:
                btype = get_block_type(block)
                if btype in ("image", "document"):
                    message_dropped_attachment = True
                    any_stripped = True
                    continue

                if btype == "tool_result":
                    inner = get_block_attr(block, "content")
                    if isinstance(inner, list):
                        filtered_inner = []
                        inner_dropped_attachment = False
                        for sub in inner:
                            sub_type = get_block_type(sub)
                            if sub_type in ("image", "document"):
                                inner_dropped_attachment = True
                                any_stripped = True
                                continue
                            filtered_inner.append(sub)

                        if not filtered_inner and inner_dropped_attachment:
                            # Replaced with placeholder
                            placeholder_text = _OMITTED_ATTACHMENT_TEXT
                            # If we explicitly had a document, we can use a more specific placeholder
                            has_doc = any(get_block_type(sub) == "document" for sub in inner)
                            has_img = any(get_block_type(sub) == "image" for sub in inner)
                            if has_doc and not has_img:
                                placeholder_text = _OMITTED_DOCUMENT_TEXT
                            
                            # Replace tool result content with placeholder text block
                            new_block = copy(block)
                            new_block.content = [{"type": "text", "text": placeholder_text}]
                            new_content.append(new_block)
                        else:
                            new_block = copy(block)
                            new_block.content = filtered_inner
                            new_content.append(new_block)
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)

            if not new_content and message_dropped_attachment:
                # Top level block list became empty
                has_doc = any(get_block_type(block) == "document" for block in content)
                has_img = any(get_block_type(block) == "image" for block in content)
                placeholder_text = _OMITTED_ATTACHMENT_TEXT
                if has_doc and not has_img:
                    placeholder_text = _OMITTED_DOCUMENT_TEXT
                new_content = [{"type": "text", "text": placeholder_text}]

            new_msg.content = new_content

        cleaned_messages.append(new_msg)

    return cleaned_messages, any_stripped


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from an Anthropic request for DeepSeek."""
    logger.debug(
        "DEEPSEEK_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    # First validate request structure (MCP servers, listed tools, etc.)
    _validate_deepseek_request_data(request_data)

    # Copy request data to clean messages safely
    cleaned_request_data = copy(request_data)
    if hasattr(cleaned_request_data, "messages") and cleaned_request_data.messages:
        cleaned_msgs, any_stripped = _clean_unsupported_blocks(cleaned_request_data.messages)
        cleaned_request_data.messages = cleaned_msgs
        if any_stripped:
            logger.warning(
                "DEEPSEEK_REQUEST: stripped unsupported attachment blocks. "
                "DeepSeek has no vision/document support; the model will not see this content."
            )

    try:
        body = build_base_request_body(
            cleaned_request_data,
            reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
            if thinking_enabled
            else ReasoningReplayMode.DISABLED,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    request_extra = getattr(cleaned_request_data, "extra_body", None)
    if isinstance(request_extra, dict) and request_extra:
        # DeepSeek doesn't accept extra top-level fields unless passed to extra_body
        body["extra_body"] = dict(request_extra)

    logger.debug(
        "DEEPSEEK_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        body.get("tools"),
    )
    return body

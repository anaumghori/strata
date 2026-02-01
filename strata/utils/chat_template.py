from typing import TypedDict
import jinja2
import jinja2.sandbox


class ChatMessage(TypedDict, total=False):
    """Standard chat message format compatible with OpenAI API."""

    role: str
    content: str
    name: str


CHATML_TEMPLATE = """{%- for message in messages -%}
{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] -}}
{%- if (loop.last and add_generation_prompt) or not loop.last -%}
{{- '<|im_end|>' + '\\n' -}}
{%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}
{{- '<|im_start|>assistant\\n' -}}
{%- endif -%}"""


_JINJA_ENV = jinja2.sandbox.ImmutableSandboxedEnvironment(
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render_jinja_template(
    template_str: str,
    messages: list[ChatMessage],
    add_generation_prompt: bool = True,
    **kwargs,
) -> str:
    """
    :param template_str: Jinja2 template string
    :param messages: List of chat messages
    :param add_generation_prompt: Whether to add assistant prompt at end
    :param kwargs: Additional template variables
    :returns: Rendered prompt string
    """
    template = _JINJA_ENV.from_string(template_str)
    return template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def apply_chat_template(
    tokenizer,
    messages: list[ChatMessage],
    tokenize: bool = False,
    add_generation_prompt: bool = True,
    chat_template: str | None = None,
) -> str | list[int]:
    """Apply chat template to messages, with fallback for models without templates.

    :param tokenizer: HuggingFace tokenizer instance
    :param messages: List of chat messages with role and content
    :param tokenize: If True, return token IDs instead of string
    :param add_generation_prompt: Whether to add assistant generation prompt
    :param chat_template: Optional explicit template to use
    :returns: Formatted prompt string or token IDs
    """
    template_to_use = chat_template

    if template_to_use is None:
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            try:
                result = tokenizer.apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                )
                return result
            except Exception:
                pass

        template_to_use = CHATML_TEMPLATE

    rendered = _render_jinja_template(
        template_to_use,
        messages,
        add_generation_prompt=add_generation_prompt,
    )

    if tokenize:
        return tokenizer.encode(rendered, add_special_tokens=False)

    return rendered


def format_messages(
    user_content: str | list[str],
    system_content: str | None = None,
) -> list[ChatMessage]:
    """Create a list of chat messages from user input.

    :param user_content: Single user message or list of user messages
    :param system_content: Optional system message to prepend
    :returns: List of ChatMessage dictionaries
    """
    messages: list[ChatMessage] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    if isinstance(user_content, str):
        messages.append({"role": "user", "content": user_content})
    else:
        for content in user_content:
            messages.append({"role": "user", "content": content})

    return messages

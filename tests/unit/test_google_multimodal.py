from types import SimpleNamespace
from unittest.mock import MagicMock

import pydantic
import pytest
from trulens.providers.google import provider as google_provider
from trulens.providers.google.provider import Google


@pytest.fixture
def mock_google(monkeypatch):
    """Create a Google provider without initializing a real Google client."""

    generate_content = MagicMock(
        return_value=SimpleNamespace(
            text="generated response",
            parsed=None,
        )
    )

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )

    # Bypass Google.__init__, which would construct a real GoogleEndpoint.
    provider = object.__new__(Google)
    object.__setattr__(provider, "model_engine", "gemini-2.5-flash")
    object.__setattr__(
        provider,
        "endpoint",
        SimpleNamespace(client=client),
    )

    # Mock Gemini SDK constructors so these tests verify our translation logic,
    # not google-genai's internal Pydantic models.
    from_text = MagicMock(
        side_effect=lambda *, text: {
            "kind": "text",
            "text": text,
        }
    )
    from_bytes = MagicMock(
        side_effect=lambda *, data, mime_type: {
            "kind": "media",
            "data": data,
            "mime_type": mime_type,
        }
    )

    monkeypatch.setattr(
        google_provider.types.Part,
        "from_text",
        from_text,
    )
    monkeypatch.setattr(
        google_provider.types.Part,
        "from_bytes",
        from_bytes,
    )

    monkeypatch.setattr(
        google_provider.types,
        "Content",
        lambda *, role, parts: {
            "role": role,
            "parts": parts,
        },
    )

    monkeypatch.setattr(
        google_provider,
        "GenerateContentConfig",
        lambda **kwargs: kwargs,
    )

    return SimpleNamespace(
        provider=provider,
        generate_content=generate_content,
        from_text=from_text,
        from_bytes=from_bytes,
    )


@pytest.mark.optional
class TestGoogleMultimodalParts:
    def test_converts_string_to_text_part(self, mock_google):
        result = mock_google.provider._to_gemini_part("Describe this image.")

        assert result == {
            "kind": "text",
            "text": "Describe this image.",
        }
        mock_google.from_text.assert_called_once_with(
            text="Describe this image."
        )
        mock_google.from_bytes.assert_not_called()

    def test_converts_text_dictionary_to_text_part(self, mock_google):
        result = mock_google.provider._to_gemini_part({
            "type": "text",
            "text": "What is shown here?",
        })

        assert result == {
            "kind": "text",
            "text": "What is shown here?",
        }
        mock_google.from_text.assert_called_once_with(
            text="What is shown here?"
        )

    def test_converts_media_dictionary_to_bytes_part(self, mock_google):
        image_data = b"\x89PNG\r\n\x1a\n"

        result = mock_google.provider._to_gemini_part({
            "type": "media",
            "data": image_data,
            "mime_type": "image/png",
        })

        assert result == {
            "kind": "media",
            "data": image_data,
            "mime_type": "image/png",
        }
        mock_google.from_bytes.assert_called_once_with(
            data=image_data,
            mime_type="image/png",
        )

    def test_media_requires_explicit_mime_type(self, mock_google):
        with pytest.raises(
            ValueError,
            match="require an explicit 'mime_type'",
        ):
            mock_google.provider._to_gemini_part({
                "type": "media",
                "data": b"image-data",
            })

        mock_google.from_bytes.assert_not_called()

    @pytest.mark.parametrize(
        "part",
        [
            123,
            None,
            ["text"],
            {"type": "video"},
            {"type": "audio", "data": b"audio"},
        ],
    )
    def test_rejects_unsupported_content_parts(
        self,
        mock_google,
        part,
    ):
        with pytest.raises(ValueError, match="Unsupported content part"):
            mock_google.provider._to_gemini_part(part)


@pytest.mark.optional
class TestGoogleMultimodalCompletion:
    def test_builds_mixed_text_and_image_message(self, mock_google):
        image_data = b"fake-png-data"

        result = mock_google.provider._create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the image.",
                        },
                        {
                            "type": "media",
                            "data": image_data,
                            "mime_type": "image/png",
                        },
                    ],
                }
            ]
        )

        assert result == "generated response"

        mock_google.generate_content.assert_called_once_with(
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Describe the image.",
                        },
                        {
                            "kind": "media",
                            "data": image_data,
                            "mime_type": "image/png",
                        },
                    ],
                }
            ],
            config={"seed": 123},
        )

    def test_preserves_multimodal_part_order(self, mock_google):
        mock_google.provider._create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Before"},
                        {
                            "type": "media",
                            "data": b"image",
                            "mime_type": "image/jpeg",
                        },
                        {"type": "text", "text": "After"},
                    ],
                }
            ]
        )

        request = mock_google.generate_content.call_args.kwargs

        assert request["contents"][0]["parts"] == [
            {
                "kind": "text",
                "text": "Before",
            },
            {
                "kind": "media",
                "data": b"image",
                "mime_type": "image/jpeg",
            },
            {
                "kind": "text",
                "text": "After",
            },
        ]

    def test_supports_multiple_user_messages(self, mock_google):
        mock_google.provider._create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "First question",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "media",
                            "data": b"image",
                            "mime_type": "image/webp",
                        }
                    ],
                },
            ]
        )

        request = mock_google.generate_content.call_args.kwargs

        assert request["contents"] == [
            {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "First question",
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "kind": "media",
                        "data": b"image",
                        "mime_type": "image/webp",
                    }
                ],
            },
        ]

    def test_adds_system_instruction_to_config(self, mock_google):
        mock_google.provider._create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Be concise.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "media",
                            "data": b"image",
                            "mime_type": "image/png",
                        }
                    ],
                },
            ]
        )

        request = mock_google.generate_content.call_args.kwargs

        assert request["config"] == {
            "seed": 123,
            "system_instruction": "Be concise.",
        }

        # The system message should not become normal Gemini content.
        assert len(request["contents"]) == 1
        assert request["contents"][0]["role"] == "user"

    def test_uses_custom_seed(self, mock_google):
        mock_google.provider._create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            seed=456,
        )

        request = mock_google.generate_content.call_args.kwargs

        assert request["config"]["seed"] == 456

    def test_prompt_path_still_works(self, mock_google):
        result = mock_google.provider._create_chat_completion(
            prompt="Text-only prompt"
        )

        assert result == "generated response"

        mock_google.generate_content.assert_called_once_with(
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Text-only prompt",
                        }
                    ],
                }
            ],
            config={"seed": 123},
        )

    def test_requires_prompt_or_messages(self, mock_google):
        with pytest.raises(
            ValueError,
            match="`prompt` or `messages` must be specified",
        ):
            mock_google.provider._create_chat_completion()


@pytest.mark.optional
class TestGoogleMultimodalStructuredOutput:
    class ImageDescription(pydantic.BaseModel):
        description: str
        object_count: int

    def test_returns_parsed_structured_response(
        self,
        mock_google,
        monkeypatch,
    ):
        parsed = self.ImageDescription(
            description="A landscape",
            object_count=2,
        )
        mock_google.generate_content.return_value = SimpleNamespace(
            text='{"description": "A landscape", "object_count": 2}',
            parsed=parsed,
        )

        monkeypatch.setattr(
            mock_google.provider,
            "_structured_output_supported",
            lambda: True,
        )

        result = mock_google.provider._create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image.",
                        },
                        {
                            "type": "media",
                            "data": b"image",
                            "mime_type": "image/png",
                        },
                    ],
                }
            ],
            response_format=self.ImageDescription,
        )

        assert result == parsed

        request = mock_google.generate_content.call_args.kwargs
        assert request["config"] == {
            "response_mime_type": "application/json",
            "response_schema": self.ImageDescription,
        }

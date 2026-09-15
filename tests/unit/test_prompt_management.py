"""Tests for connector-backed prompt management (issue #2703)."""

import tempfile
import threading
import time
import unittest

import pandas as pd
from trulens.core import TruSession
from trulens.core import prompt as core_prompt
from trulens.core.schema import prompt as prompt_schema


def _chat_messages(system: str = "Answer using the support policy."):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "{{question}}"},
    ]


class TestPromptSchema(unittest.TestCase):
    """Content, validation, and hashing, with no database involved."""

    def test_prompt_id_is_stable_across_metadata_edits(self):
        first = prompt_schema.Prompt(slug="support", name="Support")
        second = prompt_schema.Prompt(
            slug="support", name="Renamed", description="new", tags=["a"]
        )

        self.assertEqual(first.prompt_id, second.prompt_id)

    def test_prompt_id_differs_by_slug(self):
        self.assertNotEqual(
            prompt_schema.Prompt(slug="a").prompt_id,
            prompt_schema.Prompt(slug="b").prompt_id,
        )

    def test_text_and_chat_versions_hash_canonically(self):
        text = prompt_schema.PromptVersion(
            prompt_id="p", prompt_type="text", text="Answer {{question}}"
        )
        same_text = prompt_schema.PromptVersion(
            prompt_id="p",
            prompt_type="text",
            text="Answer {{question}}",
            change_note="a different note",
        )
        other_text = prompt_schema.PromptVersion(
            prompt_id="p", prompt_type="text", text="Answer {{q}}"
        )

        self.assertEqual(text.version_id, same_text.version_id)
        self.assertEqual(text.content_hash, same_text.content_hash)
        self.assertNotEqual(text.version_id, other_text.version_id)

        chat = prompt_schema.PromptVersion(
            prompt_id="p", prompt_type="chat", messages=_chat_messages()
        )
        same_chat = prompt_schema.PromptVersion(
            prompt_id="p", prompt_type="chat", messages=_chat_messages()
        )
        self.assertEqual(chat.version_id, same_chat.version_id)
        self.assertNotEqual(chat.version_id, text.version_id)

    def test_version_id_covers_defaults_and_response_format(self):
        base = prompt_schema.PromptVersion(
            prompt_id="p", prompt_type="text", text="hi"
        )
        with_defaults = prompt_schema.PromptVersion(
            prompt_id="p",
            prompt_type="text",
            text="hi",
            model_defaults={"temperature": 0.2},
        )
        with_format = prompt_schema.PromptVersion(
            prompt_id="p",
            prompt_type="text",
            text="hi",
            response_format={"type": "json_object"},
        )

        self.assertNotEqual(base.version_id, with_defaults.version_id)
        self.assertNotEqual(base.version_id, with_format.version_id)
        self.assertNotEqual(with_defaults.version_id, with_format.version_id)

    def test_version_is_immutable(self):
        version = prompt_schema.PromptVersion(
            prompt_id="p", prompt_type="text", text="hi"
        )

        with self.assertRaises(Exception):
            version.text = "bye"

    def test_content_must_match_prompt_type(self):
        with self.assertRaises(ValueError):
            prompt_schema.PromptVersion(
                prompt_id="p", prompt_type="chat", text="hi"
            )
        with self.assertRaises(ValueError):
            prompt_schema.PromptVersion(
                prompt_id="p", prompt_type="text", messages=_chat_messages()
            )

    def test_variables_are_parsed_in_order(self):
        version = prompt_schema.PromptVersion(
            prompt_id="p",
            prompt_type="text",
            text="{{ b }} then {{a}} then {{b}}",
        )

        self.assertEqual(version.variables, ["b", "a"])

    def test_undeclared_variables_are_rejected(self):
        with self.assertRaises(prompt_schema.VariableError):
            prompt_schema.PromptVersion(
                prompt_id="p",
                prompt_type="text",
                text="{{a}} {{b}}",
                variables=["a"],
            )

    def test_reserved_variable_names_are_rejected(self):
        with self.assertRaises(prompt_schema.VariableError):
            prompt_schema.PromptVersion(
                prompt_id="p", prompt_type="text", text="{{strict}}"
            )

    def test_executable_template_syntax_is_rejected(self):
        for template in (
            "{% for x in y %}{{x}}{% endfor %}",
            "{{ question | upper }}",
            "{{ __import__('os') }}",
            "{{ 1 + 1 }}",
        ):
            with self.subTest(template=template):
                with self.assertRaises(prompt_schema.InvalidTemplateError):
                    prompt_schema.PromptVersion(
                        prompt_id="p", prompt_type="text", text=template
                    )

    def test_structured_content_blocks_are_supported(self):
        version = prompt_schema.PromptVersion(
            prompt_id="p",
            prompt_type="chat",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "{{question}}"}],
                }
            ],
        )
        resolved = prompt_schema.ResolvedPrompt(
            prompt=prompt_schema.Prompt(slug="s", prompt_type="chat"),
            version=version,
        )

        rendered = resolved.render(question="why")

        self.assertEqual(version.variables, ["question"])
        self.assertEqual(
            rendered.messages,
            [{"role": "user", "content": [{"type": "text", "text": "why"}]}],
        )


class TestRendering(unittest.TestCase):
    def setUp(self):
        self.prompt = prompt_schema.Prompt(slug="support", prompt_type="chat")
        self.version = prompt_schema.PromptVersion(
            prompt_id=self.prompt.prompt_id,
            prompt_type="chat",
            messages=_chat_messages(),
            variables=["question"],
            model_defaults={"temperature": 0.2, "top_p": 0.9},
            response_format={"type": "json_object"},
        )
        self.resolved = prompt_schema.ResolvedPrompt(
            prompt=self.prompt, version=self.version, label="production"
        )

    def test_render_returns_ordered_messages(self):
        rendered = self.resolved.render(question="How do I reset MFA?")

        self.assertIsNone(rendered.text)
        self.assertEqual(
            rendered.messages,
            [
                {
                    "role": "system",
                    "content": "Answer using the support policy.",
                },
                {"role": "user", "content": "How do I reset MFA?"},
            ],
        )
        self.assertEqual(rendered.label, "production")
        self.assertEqual(rendered.version_id, self.version.version_id)

    def test_render_reports_missing_variables(self):
        with self.assertRaises(prompt_schema.VariableError) as ctx:
            self.resolved.render()

        self.assertIn("question", str(ctx.exception))

    def test_render_reports_unexpected_variables_in_strict_mode(self):
        with self.assertRaises(prompt_schema.VariableError) as ctx:
            self.resolved.render(question="a", extra="b")

        self.assertIn("extra", str(ctx.exception))

    def test_non_strict_render_drops_unexpected_values(self):
        rendered = self.resolved.render(strict=False, question="a", extra="b")

        self.assertEqual(rendered.messages[1]["content"], "a")

    def test_overrides_merge_without_mutating_defaults(self):
        rendered = self.resolved.render(
            model_overrides={"temperature": 0.0}, question="a"
        )

        self.assertEqual(
            rendered.model_args, {"temperature": 0.0, "top_p": 0.9}
        )
        self.assertEqual(
            self.version.model_defaults, {"temperature": 0.2, "top_p": 0.9}
        )

    def test_response_format_is_returned_separately(self):
        rendered = self.resolved.render(question="a")

        self.assertNotIn("type", rendered.model_args)
        self.assertEqual(rendered.response_format, {"type": "json_object"})

    def test_rendered_hash_tracks_the_rendered_values(self):
        first = self.resolved.render(question="a")
        again = self.resolved.render(question="a")
        different = self.resolved.render(question="b")

        self.assertEqual(
            first.rendered_content_hash, again.rendered_content_hash
        )
        self.assertNotEqual(
            first.rendered_content_hash, different.rendered_content_hash
        )

    def test_build_returns_messages_for_chat_only(self):
        self.assertEqual(
            self.resolved.build(question="a"),
            self.resolved.render(question="a").messages,
        )

        text_resolved = prompt_schema.ResolvedPrompt(
            prompt=prompt_schema.Prompt(slug="t"),
            version=prompt_schema.PromptVersion(
                prompt_id="t", prompt_type="text", text="hi"
            ),
        )
        with self.assertRaises(ValueError):
            text_resolved.build()


class TestLabelCache(unittest.TestCase):
    def test_disabled_by_default(self):
        cache = core_prompt.LabelCache()
        cache.set("p", "production", "v1")

        self.assertIsNone(cache.get("p", "production"))

    def test_entries_expire(self):
        cache = core_prompt.LabelCache(ttl=0.05)
        cache.set("p", "production", "v1")

        self.assertEqual(cache.get("p", "production"), "v1")
        time.sleep(0.06)
        self.assertIsNone(cache.get("p", "production"))

    def test_invalidate_scopes(self):
        cache = core_prompt.LabelCache(ttl=60)
        cache.set("p", "production", "v1")
        cache.set("p", "staging", "v2")
        cache.set("q", "production", "v3")

        cache.invalidate("p", "production")
        self.assertIsNone(cache.get("p", "production"))
        self.assertEqual(cache.get("p", "staging"), "v2")

        cache.invalidate("p")
        self.assertIsNone(cache.get("p", "staging"))
        self.assertEqual(cache.get("q", "production"), "v3")

        cache.invalidate()
        self.assertIsNone(cache.get("q", "production"))


class TestPromptPersistence(unittest.TestCase):
    """The workflow against SQLite, with no hosted service."""

    def setUp(self):
        self.db_file = tempfile.mktemp(suffix=".sqlite")
        self.session = TruSession(database_url=f"sqlite:///{self.db_file}")
        self.session.reset_database()
        self.prompt = self.session.create_prompt(
            slug="support-assistant",
            name="Support assistant",
            prompt_type="chat",
        )

    def _version(self, system: str, **kwargs):
        return self.session.create_prompt_version(
            prompt=self.prompt,
            messages=_chat_messages(system),
            variables=["question"],
            **kwargs,
        )

    def test_prompt_round_trips(self):
        stored = self.session.connector.db.get_prompt(slug="support-assistant")

        self.assertEqual(stored, self.prompt)

    def test_prompt_type_is_fixed_after_creation(self):
        with self.assertRaises(ValueError):
            self.session.create_prompt(
                slug="support-assistant", prompt_type="text"
            )

    def test_prompt_metadata_updates_in_place(self):
        self.session.create_prompt(
            slug="support-assistant",
            name="Renamed",
            prompt_type="chat",
            tags=["support"],
        )

        stored = self.session.connector.db.get_prompt(slug="support-assistant")
        self.assertEqual(stored.prompt_id, self.prompt.prompt_id)
        self.assertEqual(stored.name, "Renamed")
        self.assertEqual(stored.tags, ["support"])

    def test_creating_the_same_version_is_idempotent(self):
        first = self._version("Answer.", change_note="one")
        second = self._version("Answer.", change_note="two")

        self.assertEqual(first.version_id, second.version_id)
        self.assertEqual(len(self.session.get_prompt_versions(self.prompt)), 1)

    def test_creating_a_version_moves_latest_and_sets_parent(self):
        first = self._version("Answer.")
        second = self._version("Answer. Cite the policy.")

        self.assertEqual(second.parent_version_id, first.version_id)
        self.assertEqual(
            self.session.get_prompt(self.prompt).version_id,
            second.version_id,
        )

    def test_exact_lookup_is_stable_after_labels_move(self):
        first = self._version("Answer.")
        second = self._version("Answer. Cite the policy.")

        self.session.set_prompt_label(self.prompt, "production", first)
        self.session.set_prompt_label(self.prompt, "production", second)

        resolved = self.session.get_prompt(
            self.prompt, version_id=first.version_id
        )
        self.assertEqual(resolved.version_id, first.version_id)
        self.assertIsNone(resolved.label)
        self.assertEqual(
            resolved.render(question="q").messages[0]["content"],
            "Answer.",
        )

    def test_rollback_moves_the_label_without_touching_versions(self):
        first = self._version("Answer.")
        second = self._version("Answer. Cite the policy.")

        self.session.set_prompt_label(self.prompt, "production", second)
        self.session.set_prompt_label(
            self.prompt, "production", first, moved_by="rollback"
        )

        # Application code is unchanged: it still resolves the same label.
        resolved = self.session.get_prompt(self.prompt, label="production")
        self.assertEqual(resolved.version_id, first.version_id)

        stored_second = self.session.connector.db.get_prompt_version(
            second.version_id
        )
        self.assertEqual(stored_second, second)

    def test_label_moves_append_history(self):
        first = self._version("Answer.")
        second = self._version("Answer. Cite the policy.")

        self.session.set_prompt_label(
            self.prompt, "production", first, moved_by="ci"
        )
        self.session.set_prompt_label(
            self.prompt, "production", second, moved_by="ci"
        )

        history = self.session.get_prompt_label_history(
            self.prompt, label="production"
        )
        self.assertEqual(len(history), 2)
        self.assertEqual(
            list(history["new_version_id"]),
            [second.version_id, first.version_id],
        )
        self.assertEqual(
            history.iloc[0]["previous_version_id"], first.version_id
        )
        self.assertTrue(pd.isna(history.iloc[1]["previous_version_id"]))

    def test_concurrent_label_moves_keep_one_pointer(self):
        versions = [self._version(f"Answer {i}.") for i in range(6)]
        barrier = threading.Barrier(len(versions))
        errors = []

        def move(version):
            try:
                barrier.wait()
                self.session.set_prompt_label(
                    self.prompt, "production", version
                )
            except Exception as e:  # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=move, args=(v,)) for v in versions]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])

        labels = self.session.get_prompt_labels(self.prompt)
        production = labels[labels["label"] == "production"]
        self.assertEqual(len(production), 1)
        self.assertIn(
            production.iloc[0]["version_id"],
            {v.version_id for v in versions},
        )

        history = self.session.get_prompt_label_history(
            self.prompt, label="production"
        )
        self.assertEqual(len(history), len(versions))

    def test_cache_expires_and_can_be_bypassed(self):
        first = self._version("Answer.")
        second = self._version("Answer. Cite the policy.")
        self.session.set_prompt_label(self.prompt, "production", first)

        self.session.prompt_label_cache.ttl = 60
        self.assertEqual(
            self.session.get_prompt(self.prompt, label="production").version_id,
            first.version_id,
        )

        # Move the label behind the session's back so only the cache is stale.
        self.session.connector.db.set_prompt_label(
            prompt_id=self.prompt.prompt_id,
            label="production",
            version_id=second.version_id,
        )

        self.assertEqual(
            self.session.get_prompt(self.prompt, label="production").version_id,
            first.version_id,
        )
        self.assertEqual(
            self.session.get_prompt(
                self.prompt, label="production", use_cache=False
            ).version_id,
            second.version_id,
        )

        self.session.prompt_label_cache.ttl = 0.05
        time.sleep(0.06)
        self.assertEqual(
            self.session.get_prompt(self.prompt, label="production").version_id,
            second.version_id,
        )

    def test_unknown_label_and_version_raise(self):
        self._version("Answer.")

        with self.assertRaises(ValueError):
            self.session.get_prompt(self.prompt, label="production")
        with self.assertRaises(ValueError):
            self.session.get_prompt(self.prompt, version_id="nope")
        with self.assertRaises(ValueError):
            self.session.get_prompt("no-such-slug")

    def test_label_must_point_at_a_version_of_the_same_prompt(self):
        other = self.session.create_prompt(slug="other", prompt_type="text")
        other_version = self.session.create_prompt_version(
            prompt=other, text="hello"
        )

        with self.assertRaises(ValueError):
            self.session.set_prompt_label(
                self.prompt, "production", other_version
            )

    def test_prompt_definitions_hold_no_credentials_or_user_values(self):
        version = self._version("Answer.", model_defaults={"temperature": 0.2})
        resolved = self.session.get_prompt(
            self.prompt, version_id=version.version_id
        )
        resolved.render(question="my account number is 12345")

        stored = self.session.connector.db.get_prompt_version(
            version.version_id
        )
        serialized = stored.model_dump_json()
        self.assertNotIn("12345", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertEqual(stored.model_defaults, {"temperature": 0.2})

    def test_prompts_and_versions_are_listed(self):
        version = self._version("Answer.", change_note="one")

        prompts = self.session.get_prompts()
        self.assertEqual(list(prompts["slug"]), ["support-assistant"])
        self.assertEqual(list(prompts["prompt_type"]), ["chat"])

        versions = self.session.get_prompt_versions(self.prompt)
        self.assertEqual(list(versions["version_id"]), [version.version_id])
        self.assertEqual(list(versions["change_note"]), ["one"])
        self.assertEqual(list(versions["variables"]), [["question"]])


if __name__ == "__main__":
    unittest.main()

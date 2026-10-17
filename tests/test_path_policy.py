import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from tool_path_policy import find_relative_path_policy_violations, format_path_policy_error


class PathPolicyTests(unittest.TestCase):
    def test_url_fstring_fragments_are_not_file_paths(self):
        for code in (
            'url = f"https://pypi.org/pypi/pkg/{version}/json"',
            'url = f"http://example.com/{section}/../{page}"',
            'url = f"https://example.com/{dataset}/resolve/main/{file}"',
            'url = f"https://example.com/{section}/~user/{page}"',
        ):
            with self.subTest(code=code):
                self.assertEqual(find_relative_path_policy_violations(code), [])

    def test_tilde_character_is_valid_text(self):
        for code in (
            "text = text.replace('~', ' ')",
            "chars = ['&', '%', '^', '~']",
        ):
            with self.subTest(code=code):
                self.assertEqual(find_relative_path_policy_violations(code), [])

    def test_absolute_and_parent_paths_remain_blocked(self):
        for value in ("/json", "/tmp/a", "/mnt/data/a", "~/a", "~user/a", "../a", "pages/../../a"):
            with self.subTest(value=value):
                violations = find_relative_path_policy_violations(f"open({value!r})")
                self.assertEqual([v.value for v in violations], [value])

    def test_file_operations_inside_url_interpolations_remain_checked(self):
        for code in (
            "url = f\"https://example.com/{open('/tmp/a').read()}/json\"",
            "url = f\"https://example.com/{Path('../a').read_text()}/json\"",
            "url = f\"https://example.com/{value:{open('/tmp/a').read()}}/json\"",
        ):
            with self.subTest(code=code):
                violations = find_relative_path_policy_violations(code)
                self.assertEqual(len(violations), 1)
                self.assertIn(violations[0].value, ("/tmp/a", "../a"))

    def test_url_exemption_does_not_hide_other_literals(self):
        code = 'url = f"https://example.com/{version}/json"\nopen("/json")'
        self.assertEqual([v.value for v in find_relative_path_policy_violations(code)], ["/json"])
        for code in ('path = f"/tmp/{name}"', 'path = f"{base}/json"'):
            with self.subTest(code=code):
                self.assertTrue(find_relative_path_policy_violations(code))

    def test_relative_paths_and_plain_urls_remain_allowed(self):
        code = 'open("pages/content.md")\nurl = "https://example.com/json"'
        self.assertEqual(find_relative_path_policy_violations(code), [])

    def test_error_reports_facts_without_instructions(self):
        violations = find_relative_path_policy_violations('open("/tmp/a")')
        for label in ("Python Interpreter", "Visual Python Interpreter"):
            with self.subTest(label=label):
                message = format_path_policy_error(label, violations)
                self.assertIn(f"[{label} Error]", message)
                self.assertIn("blocked this code before execution", message)
                self.assertIn("'/tmp/a': absolute paths are not allowed", message)
                self.assertNotIn("Use relative", message)
                self.assertNotIn("Do not", message)


if __name__ == "__main__":
    unittest.main()

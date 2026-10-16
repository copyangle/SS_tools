import tempfile
import unittest
from pathlib import Path

from ss_tools.duck_safety import (
    validate_payload_extension,
    write_recovered_bytes,
)


class PayloadSafetyTests(unittest.TestCase):
    def test_known_format_extensions_are_accepted(self):
        for extension in ("png", "txt", "mp4.binpng", "mov.binpng"):
            with self.subTest(extension=extension):
                self.assertEqual(
                    validate_payload_extension(extension),
                    extension,
                )

    def test_path_components_are_rejected(self):
        invalid = (
            "",
            ".txt",
            "py",
            "exe",
            "command",
            "sh",
            "../outside",
            "folder/file",
            r"..\\outside",
            "a" * 17,
            "mp4.binpng/../../outside",
        )
        for extension in invalid:
            with self.subTest(extension=extension):
                with self.assertRaises(ValueError):
                    validate_payload_extension(extension)

    def test_output_names_are_unique_and_existing_data_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(write_recovered_bytes(directory, b"first", ".txt"))
            second = Path(write_recovered_bytes(directory, b"second", ".txt"))

            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b"first")
            self.assertEqual(second.read_bytes(), b"second")
            self.assertTrue(first.name.startswith("duck_recovered_"))
            self.assertTrue(second.name.startswith("duck_recovered_"))


    def test_unsafe_output_suffixes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".exe", ".command", ".sh", "../outside"):
                with self.subTest(suffix=suffix):
                    with self.assertRaises(ValueError):
                        write_recovered_bytes(directory, b"x", suffix)


if __name__ == "__main__":
    unittest.main()

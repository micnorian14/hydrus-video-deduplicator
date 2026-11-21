from __future__ import annotations

import unittest

from hydrusvideodeduplicator.dedup_util import display_path

WIN_ENV = {"LOCALAPPDATA": "C:\\Users\\Jane Doe\\AppData\\Local", "USERPROFILE": "C:\\Users\\Jane Doe"}
DB = "C:\\Users\\Jane Doe\\AppData\\Local\\hydrusvideodeduplicator\\hydrusvideodeduplicator\\videohashes.sqlite"


class TestDisplayPath(unittest.TestCase):
    def win(self, path, env=WIN_ENV):
        return display_path(path, windows=True, environ=env)

    def test_windows_localappdata(self):
        self.assertEqual(
            self.win(DB), "%LOCALAPPDATA%\\hydrusvideodeduplicator\\hydrusvideodeduplicator\\videohashes.sqlite"
        )

    def test_windows_case_insensitive_and_trailing_slash(self):
        env = {"LOCALAPPDATA": "c:\\users\\JANE DOE\\appdata\\local\\"}
        self.assertEqual(self.win(DB, env), "%LOCALAPPDATA%\\hydrusvideodeduplicator\\hydrusvideodeduplicator\\videohashes.sqlite")

    def test_windows_userprofile_fallback(self):
        self.assertEqual(self.win("C:\\Users\\Jane Doe\\projects\\.env"), "%USERPROFILE%\\projects\\.env")
        self.assertEqual(self.win(DB, {"USERPROFILE": "C:\\Users\\Jane Doe"}),
                         "%USERPROFILE%\\AppData\\Local\\hydrusvideodeduplicator\\hydrusvideodeduplicator\\videohashes.sqlite")

    def test_windows_other_paths_unchanged(self):
        for p in ("T:\\hvd\\db\\videohashes.sqlite", "C:\\Users\\Jane Doer\\x.sqlite", "", "relative\\x"):
            self.assertEqual(self.win(p), p)
        self.assertEqual(self.win(DB, {}), DB)  # nothing set: unchanged
        self.assertEqual(self.win(DB, {"USERPROFILE": "C:"}), DB)  # a bare drive is never masked

    def test_path_objects(self):
        from pathlib import PureWindowsPath
        self.assertTrue(self.win(PureWindowsPath(DB)).startswith("%LOCALAPPDATA%\\"))

    def test_posix_home(self):
        home = "/home/jane"
        p = "/home/jane/.local/share/hydrusvideodeduplicator/videohashes.sqlite"
        self.assertEqual(display_path(p, windows=False, home=home), "~/.local/share/hydrusvideodeduplicator/videohashes.sqlite")
        self.assertEqual(display_path("/home/janet/x", windows=False, home=home), "/home/janet/x")
        self.assertEqual(display_path("/srv/db", windows=False, home=home), "/srv/db")
        self.assertEqual(display_path("/srv/db", windows=False, home="/"), "/srv/db")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from hydrusvideodeduplicator.dedup_util import display_path

WIN_ENV = {"LOCALAPPDATA": "C:\\Users\\Jane Doe\\AppData\\Local", "USERPROFILE": "C:\\Users\\Jane Doe"}
DB = "C:\\Users\\Jane Doe\\AppData\\Local\\hydrusvideodeduplicator\\hydrusvideodeduplicator\\videohashes.sqlite"


class TestDisplayPath(unittest.TestCase):
    def win(self, path, env=WIN_ENV):
        return display_path(path, windows=True, environ=env)

    def test_windows_standard_localappdata_shown_under_userprofile(self):
        self.assertEqual(
            self.win(DB),
            "%USERPROFILE%\\AppData\\Local\\hydrusvideodeduplicator\\hydrusvideodeduplicator\\videohashes.sqlite",
        )
        # LOCALAPPDATA equal to USERPROFILE\AppData\Local in a different case still counts
        env = {"LOCALAPPDATA": "c:\\users\\jane doe\\appdata\\local", "USERPROFILE": "C:\\Users\\Jane Doe\\"}
        self.assertEqual(
            self.win(DB, env),
            "%USERPROFILE%\\AppData\\Local\\hydrusvideodeduplicator\\hydrusvideodeduplicator\\videohashes.sqlite",
        )

    def test_windows_moved_localappdata_keeps_localappdata(self):
        env = {"LOCALAPPDATA": "D:\\Profiles\\Jane\\Local", "USERPROFILE": "C:\\Users\\Jane Doe"}
        self.assertEqual(
            self.win("D:\\Profiles\\Jane\\Local\\hydrusvideodeduplicator\\videohashes.sqlite", env),
            "%LOCALAPPDATA%\\hydrusvideodeduplicator\\videohashes.sqlite",
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
        self.assertTrue(self.win(PureWindowsPath(DB)).startswith("%USERPROFILE%\\AppData\\Local\\"))

    def test_posix_home(self):
        home = "/home/jane"
        p = "/home/jane/.local/share/hydrusvideodeduplicator/videohashes.sqlite"
        self.assertEqual(display_path(p, windows=False, home=home), "~/.local/share/hydrusvideodeduplicator/videohashes.sqlite")
        self.assertEqual(display_path("/home/janet/x", windows=False, home=home), "/home/janet/x")
        self.assertEqual(display_path("/srv/db", windows=False, home=home), "/srv/db")
        self.assertEqual(display_path("/srv/db", windows=False, home="/"), "/srv/db")


if __name__ == "__main__":
    unittest.main()

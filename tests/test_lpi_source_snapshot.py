from pathlib import Path
import subprocess
import os
import tarfile
import tempfile
import unittest
from tools.lpi_source_snapshot import capture_source,persist_preflight
from tools.lpi_common import read_json,write_json


class SourceSnapshotTests(unittest.TestCase):
    def test_snapshot_preserves_modified_untracked_and_ignores_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'repo';root.mkdir()
            def git(*args):
                subprocess.run(['git',*args],cwd=root,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            git('init');git('config','user.name','Test');git('config','user.email','test@example.invalid')
            (root/'.gitignore').write_text('data/\n')
            (root/'code.py').write_text('old\n');git('add','.');git('commit','-m','initial')
            (root/'code.py').write_text('new\n');(root/'new.py').write_text('untracked\n')
            (root/'data').mkdir();(root/'data/ignored.bin').write_bytes(b'dataset')
            captured=capture_source(Path(tmp)/'snapshot',root=root)
            with tarfile.open(Path(captured['directory'])/'source.tar.gz') as archive:
                self.assertEqual(archive.extractfile('code.py').read(),b'new\n')
                self.assertEqual(archive.extractfile('new.py').read(),b'untracked\n')
                self.assertNotIn('data/ignored.bin',archive.getnames())
            run=Path(tmp)/'run';run.mkdir();preflight=Path(tmp)/'preflight.json'
            write_json(preflight,{'source_snapshot':captured,'git':{'status':'dirty'}})
            persist_preflight(preflight,run/'preflight.json')
            stored=read_json(run/'preflight.json')['source_snapshot']
            self.assertTrue(Path(stored['directory']).is_relative_to(run))
            self.assertEqual(stored['tree_sha256'],captured['tree_sha256'])
            # Same source contents with changed mtimes must still be resumable.
            stat=(root/'code.py').stat();os.utime(root/'code.py',(stat.st_atime,stat.st_mtime+5))
            again=capture_source(Path(tmp)/'snapshot-again',root=root)
            self.assertEqual(again['tree_sha256'],captured['tree_sha256'])
            write_json(Path(tmp)/'again.json',{'source_snapshot':again})
            persist_preflight(Path(tmp)/'again.json',run/'resume.json')
            # An idempotent second persistence is safe, but corrupted snapshots must fail.
            persist_preflight(preflight,run/'preflight2.json')
            (Path(stored['directory'])/'source.tar.gz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):persist_preflight(preflight,run/'bad.json')

if __name__=='__main__':unittest.main()

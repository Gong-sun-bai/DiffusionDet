"""Preserve tracked and non-ignored untracked source files for pre-commit training."""
from pathlib import Path
import shutil
import subprocess
import tarfile

from tools.lpi_common import ROOT, digest, sha256, read_json, write_json


def capture_source(output, root=ROOT):
    root=Path(root).resolve();output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    raw=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=root)
    names=sorted(set(raw.decode().split('\0'))-{''})
    files={}
    archive=output/'source.tar.gz'
    with tarfile.open(archive,'w:gz',dereference=False) as tar:
        for name in names:
            path=root/name
            if path.is_symlink():
                files[name]={'symlink':str(path.readlink())}
            elif path.is_file():
                files[name]={'sha256':sha256(path),'size':path.stat().st_size}
            else:
                continue  # Deletions are recorded by git status/diff, not resurrected.
            tar.add(path,arcname=name,recursive=False)
    patch=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root)
    (output/'working-tree.patch').write_bytes(patch)
    manifest={'files':files,'tree_sha256':digest(files),'archive_sha256':sha256(archive)}
    write_json(output/'source.json',manifest)
    return {'directory':str(output.resolve()),'tree_sha256':manifest['tree_sha256'],
            'archive_sha256':manifest['archive_sha256'],'file_count':len(files)}


def persist_preflight(source, destination):
    source=Path(source);destination=Path(destination)
    record=read_json(source)
    if 'source_snapshot' in record:
        meta=record['source_snapshot'];folder=destination.parent/'source_snapshots'/meta['archive_sha256']
        folder.parent.mkdir(exist_ok=True)
        if not folder.exists():
            shutil.copytree(meta['directory'],folder)
        if sha256(folder/'source.tar.gz')!=meta['archive_sha256']:
            raise ValueError('源码快照哈希不一致')
        record['source_snapshot']={**meta,'directory':str(folder.resolve())}
    write_json(destination,record)

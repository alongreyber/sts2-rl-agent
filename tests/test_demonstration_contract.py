import gzip,hashlib,json,tempfile,unittest
from pathlib import Path
from sts2_env.demonstrations import load_dataset

class ContractTests(unittest.TestCase):
    def test_checksum_failure_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'bad.gz').write_bytes(b'changed')
            (root/'manifest.json').write_text(json.dumps({'schema':'sts2-demonstrations/v1','files':[{'path':'bad.gz','sha256':'bad'}]}))
            with self.assertRaisesRegex(ValueError,'checksum'):load_dataset(root,'combat')
    def test_manifest_cannot_escape_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'manifest.json').write_text(json.dumps({'schema':'sts2-demonstrations/v1','files':[{'path':'../secret','sha256':'bad'}]}))
            with self.assertRaisesRegex(ValueError,'Unsafe'):load_dataset(root,'combat')

if __name__=='__main__':unittest.main()

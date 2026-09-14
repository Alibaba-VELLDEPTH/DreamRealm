"""Bound torch checkpoint object size on OSS mounts without changing tensors."""
import bisect
import hashlib
import io
import json
from pathlib import Path
import torch


class PartWriter:
    def __init__(self,root,limit):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.limit=limit;self.parts=[];self.file=None;self.size=0;self.position=0
    def _close_part(self):
        if self.file is not None:
            self.file.flush();self.file.close()
            self.parts.append(dict(name=self.name,bytes=self.size,sha256=self.digest.hexdigest()))
            self.file=None
    def write(self,data):
        view=memoryview(data);total=len(view)
        while view:
            if self.file is None:
                self.name=f'part-{len(self.parts):05d}.bin';self.file=(self.root/self.name).open('xb')
                self.size=0;self.digest=hashlib.sha256()
            n=min(len(view),self.limit-self.size);chunk=view[:n]
            written=self.file.write(chunk)
            if written!=n:raise IOError('Short checkpoint write')
            self.digest.update(chunk);self.size+=n;self.position+=n;view=view[n:]
            if self.size==self.limit:self._close_part()
        return total
    def flush(self):
        if self.file is not None:self.file.flush()
    def tell(self):return self.position
    def finish(self):
        self._close_part()
        (self.root/'index.json').write_text(json.dumps(dict(format='torch-stream-parts-v1',bytes=self.position,parts=self.parts),indent=2))


class PartReader(io.RawIOBase):
    def __init__(self,root,verify=True):
        self.root=Path(root);index=json.loads((self.root/'index.json').read_text())
        if index['format']!='torch-stream-parts-v1':raise ValueError('Unknown checkpoint format')
        self.parts=index['parts'];self.ends=[];total=0
        for part in self.parts:
            path=self.root/part['name']
            if Path(part['name']).name!=part['name']:raise ValueError('Invalid part name')
            if path.stat().st_size!=part['bytes']:raise IOError('Incomplete checkpoint part')
            if verify:
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
                if h.hexdigest()!=part['sha256']:raise IOError('Checkpoint checksum mismatch')
            total+=part['bytes'];self.ends.append(total)
        if total!=index['bytes']:raise IOError('Checkpoint length mismatch')
        self.length=total;self.position=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.length+offset if whence==2 else -1
        if target<0:raise ValueError('Invalid seek')
        self.position=target;return target
    def read(self,size=-1):
        remaining=max(0,self.length-self.position) if size<0 else min(size,max(0,self.length-self.position))
        result=bytearray()
        while remaining:
            i=bisect.bisect_right(self.ends,self.position);start=0 if i==0 else self.ends[i-1]
            n=min(remaining,self.ends[i]-self.position)
            with (self.root/self.parts[i]['name']).open('rb') as f:
                f.seek(self.position-start);chunk=f.read(n)
            if len(chunk)!=n:raise IOError('Short checkpoint read')
            result.extend(chunk);self.position+=n;remaining-=n
        return bytes(result)
    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)


def save_parts(state,root,part_bytes=1024**3):
    if part_bytes<1:raise ValueError('Invalid part size')
    writer=PartWriter(root,part_bytes)
    try:torch.save(state,writer);writer.finish()
    finally:
        if writer.file is not None:writer.file.close()


def load_parts(root):
    with PartReader(root) as reader:return torch.load(reader,map_location='cpu',weights_only=True)

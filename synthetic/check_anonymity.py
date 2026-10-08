"""Audit distributed files, tensor archives and optional PDF contents for identifiers.

Pass private names with --forbidden; they are never stored in the report.
PDF inspection requires the optional requirements-audit.txt dependency.
"""
import argparse,ast,json,re,subprocess,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EMAIL=re.compile(r'[A-Za-z0-9_.+\-]+@[A-Za-z0-9_.\-]+\.[A-Za-z]{2,}')
PERSONAL_PATH=re.compile(r'/(?:home|Users|data/user_data)/[^/\s]+')
TEXT_EXTENSIONS={'.py','.md','.tex','.json','.csv','.txt','.sha256'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forbidden',action='append',default=[],help='Private substring to reject, case-insensitively; repeatable.')
    parser.add_argument('--pdf',action='store_true',help='Inspect PDF text, metadata and attachments using pypdf.')
    parser.add_argument('--git',action='store_true',help='Require anonymous author/committer identities throughout current branch history.')
    args=parser.parse_args()
    tokens=[s.casefold() for s in args.forbidden]
    errors=[];files=archives=pdfs=0
    def inspect(label,text,check_generic=True):
        if any(t in text.casefold() for t in tokens):errors.append((label,'blocked substring'))
        if check_generic:
            if any(v!='anonymous@invalid.example' for v in EMAIL.findall(text)):errors.append((label,'email address'))
            if PERSONAL_PATH.search(text):errors.append((label,'personal filesystem path'))
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or '.git' in path.parts:continue
        relative=str(path.relative_to(ROOT));files+=1
        inspect(relative,relative)
        data=path.read_bytes()
        # Includes UTF-8 and UTF-16 strings embedded in binary formats.
        for encoding in ['utf8','utf-16-le','utf-16-be']:
            inspect(relative,data.decode(encoding,errors='ignore'),False)
        if path.suffix=='.py':
            tree=ast.parse(data.decode('utf8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Constant) and isinstance(node.value,str):
                    inspect(relative,node.value)
            import io,tokenize
            for token in tokenize.generate_tokens(io.StringIO(data.decode('utf8')).readline):
                if token.type==tokenize.COMMENT:inspect(relative,token.string)
        elif path.suffix in TEXT_EXTENSIONS:inspect(relative,data.decode('utf8'))
        if zipfile.is_zipfile(path):
            archives+=1
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    inspect(relative,member.filename)
                    raw=archive.read(member)
                    inspect(relative,raw.decode('utf8',errors='ignore'),False)
                    if member.filename.endswith('.pkl'):
                        import pickletools
                        for _,value,_ in pickletools.genops(raw):
                            if isinstance(value,str):inspect(relative,value)
        if args.pdf and path.suffix=='.pdf':
            from pypdf import PdfReader
            reader=PdfReader(path);pdfs+=1
            inspect(relative,str(reader.metadata))
            for page in reader.pages:inspect(relative,page.extract_text() or '')
            if reader.metadata or '/Metadata' in reader.trailer['/Root']:
                errors.append((relative,'document metadata present'))
            if list(reader.attachments):errors.append((relative,'embedded attachment present'))
    if args.git:
        identities=subprocess.check_output(['git','log','--format=%an|%ae|%cn|%ce','HEAD'],cwd=ROOT,text=True).splitlines()
        expected='Anonymous|anonymous@invalid.example|Anonymous|anonymous@invalid.example'
        if not identities or any(v!=expected for v in identities):errors.append(('Git history','non-anonymous commit identity'))
    result=dict(files_checked=files,tensor_archives_checked=archives,pdfs_inspected=pdfs,private_substrings_checked=len(tokens),git_history_checked=args.git,findings=[dict(file=p,kind=k) for p,k in sorted(set(errors))])
    print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()

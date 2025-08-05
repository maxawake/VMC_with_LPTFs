from pathlib import Path
import mkdocs_gen_files

PYTHON_MODULE_NAME = "tvmc"
MKDOCS_DOCS_PATH = "docs"

src_root = Path(PYTHON_MODULE_NAME)
for path in src_root.glob("**/*.py"):
    doc_path = Path(MKDOCS_DOCS_PATH, path.relative_to(src_root)).with_suffix(".md")

    with mkdocs_gen_files.open(doc_path, "w") as f:
        ident = ".".join(path.with_suffix("").parts)
        print("::: " + ident, file=f)

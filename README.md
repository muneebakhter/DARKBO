# DARKBO
Document Augmented Retrieval Knowledge Base Operator

## Prebuilding vector stores

Vector stores can be created in advance from a mapping file and a set of project
folders.  The mapping file `proj_mapping.txt` should contain project IDs and
names separated by tabs, for example:

```
95\tASPCA
175\tACLU
```

Each project folder is named after the project ID and may include `<id>.faq.json`
and/or `<id>.kb.json` files.  FAQ files contain question/answer pairs while KB
files contain article title/content pairs.

Run the prebuild script to create all vector stores:

```
python tools/prebuild_vector_stores.py
```

Vector stores are written to the path defined by `VECTOR_STORE_DIR` in
`config/settings.py`.

# active_ner

INSTALL:

1. create a `.env` file in the project root dir with a text like this:
```
ALLOW_SIGNUP=False
DATABASE_URL=postgres://username:password@host:5432/dbname?sslmode=disable
```

2. run `make setup`

Some dependencies it uses:
gensim, compress-fasttext, spacy, django

RUN:

run 
```
.venv/bin/python -m anno.learn 5
```
The script will do the following:
 - download lenta dataset from Corus (if it's not already downloaded)
 - remove up to 5 unapproved documents from the database
 - will add instead 5 annotated documents (with predicted annotations).

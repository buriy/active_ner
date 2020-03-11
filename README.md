# active_ner

INSTALL:

1. create a `.env` file in the project root dir with a text like this:
```
ALLOW_SIGNUP=False
DATABASE_URL=postgres://username:password@host:5432/dbname?sslmode=disable
```

2. run `make setup`

RUN:

run 
```
.venv/bin/python -m anno.learn
```

# Deprecation Notes

## Changes in 0.2.2

### Breaking changes to databases

- json serialization of WithClassInfo mixin
    - key "class_info" renamed to "__tru_class_info"

- json serialization of ObjSerial class
    - init_kwargs expanded into init_bindings, serialized by a new class
      Bindings, containing both args and kwargs

## Changes in 0.2.0

### Non-breaking changes to be breaking in 1.0.0

- file tru_feedback.py renamed to feedback.py

- file tru_db.py renamed to db.py
    - class TruDB renamed to DB

- file tru_app.py renamed to app.py
    - class TruApp renamed to App

- in file schema.py
    - class App renamed to AppDefinition

### Breaking changes to databases

- db schema changes

    - chain_id renamed to app_id

# Deprecation Notes

## Changes in 0.4.0

### Backwards compatible changes

- class `Record` of schema.py

    - attributes `main_input` and `main_output` generalized to `JSON` (not strict json) allowing the field to contain strings like before the change but also sequences and dictionaries (with string keys) of json.

## Changes in 0.2.3

### Backwards compatible changes

Backwards compatibility will be removed in 0.3.0 at earliest.

- class `Query` of schema.py renamed to `Select`.

## Changes in 0.2.2

### Breaking changes to databases

- json serialization of `WithClassInfo` mixin
    - key "class_info" renamed to "__tru_class_info"

- json serialization of `ObjSerial` class
    - init_kwargs expanded into init_bindings, serialized by a new class
      `Bindings`, containing both args and kwargs

## Changes in 0.2.0

### Backwards compatible changes

Backwards compatibility will be removed in 0.3.0 at earliest.

- file tru_feedback.py renamed to feedback.py

- file tru_db.py renamed to db.py
    - class `TruDB` renamed to `DB`

- file tru_app.py renamed to app.py
    - class `TruApp` renamed to `App`

- in file schema.py
    - class `App` renamed to `AppDefinition`

### Breaking changes to databases

- db schema changes

    - chain_id renamed to app_id

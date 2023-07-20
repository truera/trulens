# Deprecation Notes

## Changes in 0.5.0

### Backwards compatible

- Class `Provider` contains the attribute `endpoint` which was previously
  excluded from serialization but is now included.


## Changes in 0.4.0

### Backwards compatible

- Class `Record` of `schema.py`:

    - Attributes `main_input`, `main_output`, and `main_error` generalized to
      `JSON` (not strict json) allowing the attribute to contain strings like
      before the change but also sequences and dictionaries (with string keys)
      of json.

## Changes in 0.2.3

### Backwards compatible

Backwards compatibility will be removed in 0.3.0 at earliest.

- class `Query` of schema.py renamed to `Select` .

## Changes in 0.2.2

### Breaking changes to databases

- Json serialization of `WithClassInfo` mixin:
    - Key `class_info` renamed to `__tru_class_info` .

- Json serialization of `ObjSerial` class:
    - Attribute `init_kwargs` expanded into `init_bindings`, serialized by a new
      class `Bindings`, containing both args and kwargs.

## Changes in 0.2.0

### Backwards compatible

Backwards compatibility will be removed in 0.3.0 at earliest.

- File `tru_feedback.py` renamed to `feedback.py` .

- File `tru_db.py` renamed to `db.py` and:
    - Class `TruDB` renamed to `DB` .

- File `tru_app.py` renamed to `app.py` and:
    - Class `TruApp` renamed to `App` .

- In file `schema.py`:
    - Class `App` renamed to `AppDefinition` .

### Breaking changes to databases

- DB schema changes:

    - Table `apps`: 

        - Field `chain_id` renamed to `app_id` .

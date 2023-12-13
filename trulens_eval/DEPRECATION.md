# Deprecation Notes

## Changes in 0.19.0

- Migrated from pydantic v1 to v2 incurring various changes.
- `ObjSerial` class removed. `Obj` now indicate whether they are loadable when
  `init_bindings` is not None.
- `SingletonPerName` field `instances` renamed to `_instances` due to possible
  shadowing of `instances` field in subclassed models.
- `WithClassInfo` field `__tru_class_info` renamed to `tru_class_info`
  as pydantic does not allow underscore fields.

## Changes in 0.10.0

### Backwards compatible

- Database interfaces changed from sqlite to sqlalchemy. Sqlite databases are
  supported under the sqlaclchemy interface and other databases such as mysql
  and postgress are also now usable. Running the migration scripts via
  `Tru().migrate_database()` may be necessary.

## Changes in 0.7.0

### Backwards compatible

- Class `Cost` has new field `n_stream_chunks` to count the number of received
  chunks in streams. This is only counted when streaming mode (i.e. in OpenAI)
  is used.

## Changes in 0.6.0

### Backwards compatible

- Class `Provider` contains the attribute `endpoint` which was previously
  excluded from serialization but is now included.

- Class `FeedbackCall` has new attribute `meta` for storing additional feedback
  results. The value will be set to an empty dict if loaded from an older
  database that does not have this attribute.

- Class `FeedbackCall` has new attribute `meta` for storing additional feedback

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

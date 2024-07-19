## Pre-requisites

1. `npm` should be installed. To verify, run the following command in your terminal:

```
$ npm -v
```

If `npm` is absent, follow the instructions [here](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) to install Node.js and npm.

## Quickstart for developing

1. `cd` into the `record_viewer` directory

```
$ cd <your-trulens-directory>/trulens/trulens_eval/trulens_eval/react_components/record_viewer
```

2. Install the frontend dependencies:

```
$ npm i
```

3. Start the frontend server:

```
$ npm run dev
```

4. Set `_RELEASE` in `__init__.py` to be `False`

```
$ cd <your-trulens-directory>/trulens/trulens_eval/trulens_eval/react_components/record_viewer
$ <vi/nano/your text editor> __init__.py
```

5. Start your jupyter notebook

```
$ PYTHONPATH="<path to trulens-eval>:$PYTHONPATH" jupyter lab
```

## Quickstart once development is complete

1. `cd` into the `record_viewer` directory

```
$ cd <your-trulens-directory>/trulens/trulens_eval/trulens_eval/react_components/record_viewer
```

2. Install the frontend dependencies:

```
$ npm i
```

3. Build the files

```
$ npm run build
```

4. Set `_RELEASE` in `__init__.py` to be `True`

```
$ cd <your-trulens-directory>/trulens/trulens_eval/trulens_eval/react_components/record_viewer
$ <vi/nano/your text editor> __init__.py
```

5. Start your jupyter notebook

```
$ PYTHONPATH="<path to trulens-eval>:$PYTHONPATH" jupyter lab
```

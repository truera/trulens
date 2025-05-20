## Pre-requisites

1. `npm` should be installed. To verify, run the following command in your terminal:

```
npm -v
```

If `npm` is absent, follow the instructions [here](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) to install Node.js and npm.

## Quickstart for developing

1. `cd` into the `record_viewer` directory

```
cd <your-trulens-directory>/src/dashboard/trulens/dashboard/react_components/record_viewer
```

2. Install the frontend dependencies:

```
npm i
```

3. Start the frontend server:

```
npm run dev
```

4. Set `_RELEASE` in `__init__.py` to be `False`

```
cd <your-trulens-directory>/src/dashboard/trulens/dashboard/react_components/record_viewer
<vi/nano/your text editor> __init__.py
```

5. Start your jupyter notebook

```
PYTHONPATH="<path to trulens>:$PYTHONPATH" jupyter lab
```

## Quickstart once development is complete

1. `cd` into the `record_viewer` directory

```
cd <your-trulens-directory>/src/dashboard/trulens/dashboard/react_components/record_viewer
```

2. Install the frontend dependencies:

```
npm i
```

3. Build the files

```
npm run build
```

4. Set `_RELEASE` in `__init__.py` to be `True`

```
cd <your-trulens-directory>/src/dashboard/trulens/dashboard/react_components/record_viewer
<vi/nano/your text editor> __init__.py
```

5. Start your jupyter notebook

```
PYTHONPATH="<path to trulens>:$PYTHONPATH" jupyter lab
```

## Docker-Based Storybook Visual Testing

This project includes Docker-based visual testing for Storybook components, which helps ensure consistent snapshots across different environments and developer machines.

### Why Use Docker for Visual Testing?

- **Consistency**: Docker ensures the same rendering environment across different machines and CI systems
- **Reproducibility**: Tests produce the same results regardless of the host operating system
- **Isolation**: Testing happens in a contained environment, preventing host system variables from affecting results

### Setup

We've set up the following components for Docker-based visual testing:

1. `Dockerfile.test`: Contains the environment for running Playwright tests
2. `docker-compose.test.yml`: Configures the Docker service for testing
3. Updated npm scripts in `package.json` for Docker operations
4. Modified `takeStorySnapshot.ts` to work with static Storybook files in Docker
5. Updated `playwright.config.ts` for Docker compatibility

### Using Docker for Visual Testing

#### Running Tests

Run visual tests in Docker:

```bash
npm run test:docker-storybook:run
```

Update visual snapshots in Docker:

```bash
npm run test:docker-storybook:update
```

### How It Works

1. The Docker container builds a static version of Storybook
2. Tests run against the static build
3. Snapshots are stored in `/test/snapshots` and mounted to the Docker container
4. Test results are stored in `/test-results` and mounted from the Docker container

### Troubleshooting

- If snapshots fail with minor pixel differences, consider adjusting the `threshold` in `playwright.config.ts`
- If fonts appear different in Docker vs. your local environment, ensure fonts are consistently installed

### Best Practices

- Commit all baseline snapshots to git
- Run visual tests in Docker before submitting PRs
- Update snapshots deliberately when designs change

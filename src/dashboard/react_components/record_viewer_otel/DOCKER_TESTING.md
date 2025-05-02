# Docker-Based Storybook Visual Testing

This project includes Docker-based visual testing for Storybook components, which helps ensure consistent snapshots across different environments and developer machines.

## Why Use Docker for Visual Testing?

- **Consistency**: Docker ensures the same rendering environment across different machines and CI systems
- **Reproducibility**: Tests produce the same results regardless of the host operating system
- **Isolation**: Testing happens in a contained environment, preventing host system variables from affecting results

## Setup

We've set up the following components for Docker-based visual testing:

1. `Dockerfile.test`: Contains the environment for running Playwright tests
2. `docker-compose.test.yml`: Configures the Docker service for testing
3. Updated npm scripts in `package.json` for Docker operations
4. Modified `takeStorySnapshot.ts` to work with static Storybook files in Docker
5. Updated `playwright.config.ts` for Docker compatibility

## Using Docker for Visual Testing

### Building the Docker Image

First, build the Docker image for testing:

```bash
npm run test:docker:build
```

### Running Tests

Run visual tests in Docker:

```bash
npm run test:docker:run
```

### Updating Snapshots

When component designs change intentionally, update the baseline snapshots:

```bash
npm run test:docker:update
```

## How It Works

1. The Docker container builds a static version of Storybook
2. Tests run against the static build using `file://` protocol (no server needed)
3. Snapshots are stored in `/test/snapshots` and mounted to the Docker container
4. Test results are stored in `/test-results` and mounted from the Docker container

## Troubleshooting

- If snapshots fail with minor pixel differences, consider adjusting the `threshold` in `playwright.config.ts`
- If fonts appear different in Docker vs. your local environment, ensure fonts are consistently installed

## Best Practices

- Commit all baseline snapshots to git
- Run visual tests in Docker before submitting PRs
- Update snapshots deliberately when designs change

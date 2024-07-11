## Pre-requisites

1. Set up [snow cli](https://docs.snowflake.com/en/developer-guide/snowflake-cli-v2/index)
2. Set up a cortex search service.
3. Your role should have permissions to create native applications.

**If not running on mlplatformtest**:

1. If necessary, update the code in `snowflake.yml` to choose a different role/warehouse and run it.
2. Update the code in `pre_setup.sql` with the appropriate <role>/<database>.
3. Update the image repository in `app/manifest.yml` and `app/setup_script.sql`.
4. Update the parameters in `spcs/docker_cmds.sh` - this determines the image repository.

## Steps

1. Build the image

```
$ cd <trulens repo>/trulens_eval/examples/snowflake_demo
$ ./spcs/docker_cmds.sh
```

2. Run the app locally

```
$ snow app run
```

3. Open the link provided in the output of the previous step. It should look like:

```
Your application object (trulens_demo_na) is now available:
https://app.snowflake.com/SFENGINEERING/mlplatformtest/#/apps/application/TRULENS_DEMO_NA
```

4. Once app has started, run the code in `post_app_creation.sql` according to what you need.

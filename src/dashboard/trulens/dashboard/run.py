from __future__ import annotations

from multiprocessing import Process
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
from threading import Thread
from typing import Optional

from trulens.core import session as core_session
from trulens.core.database.connector.base import DBConnector
from trulens.core.utils import imports as import_utils
from trulens.dashboard.utils import notebook_utils
from typing_extensions import Annotated
from typing_extensions import Doc

DASHBOARD_START_TIMEOUT: Annotated[
    int, Doc("Seconds to wait for dashboard to start")
] = 30


def find_unused_port() -> int:
    """Find an unused port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _is_snowflake_connector(connector: DBConnector):
    try:
        from trulens.connectors.snowflake import SnowflakeConnector
    except ImportError:
        return False
    return isinstance(connector, SnowflakeConnector)


def run_dashboard(
    session: Optional[core_session.TruSession] = None,
    port: Optional[int] = None,
    address: Optional[str] = None,
    force: bool = False,
    sis_compatibility_mode: bool = False,
    _dev: Optional[Path] = None,
    _watch_changes: bool = False,
) -> Process:
    """Run a streamlit dashboard to view logged results and apps.

    Args:
        port (Optional[int]): Port number to pass to streamlit through `server.port`.

        address (Optional[str]): Address to pass to streamlit through `server.address`. `address` cannot be set if running from a colab notebook.

        force (bool): Stop existing dashboard(s) first. Defaults to `False`.

        sis_compatibility_mode (bool): Flag to enable compatibility with Streamlit in Snowflake (SiS). SiS runs on Python 3.8, Streamlit 1.35.0, and does not support bidirectional custom components. As a result, enabling this flag will replace custom components in the dashboard with native Streamlit components. Defaults to `False`.

        _dev (Path): If given, runs the dashboard with the given `PYTHONPATH`. This can be used to run the dashboard from outside of its pip package installation folder. Defaults to `None`.

        _watch_changes (bool): If `True`, the dashboard will watch for changes in the code and update the dashboard accordingly. Defaults to `False`.

    Returns:
        The [Process][multiprocessing.Process] executing the streamlit dashboard.

    Raises:
        RuntimeError: Dashboard is already running. Can be avoided if `force` is set.

    """
    session = session or core_session.TruSession()

    session.connector.db.check_db_revision()

    IN_COLAB = "google.colab" in sys.modules
    if IN_COLAB and address is not None:
        raise ValueError("`address` argument cannot be used in colab.")

    if force:
        stop_dashboard(force=force)

    print("Starting dashboard ...")

    # run leaderboard with subprocess
    leaderboard_path = import_utils.static_resource(
        "dashboard", "Leaderboard.py"
    )

    if session._dashboard_proc is not None:
        print("Dashboard already running at path:", session._dashboard_urls)
        return session._dashboard_proc

    env_opts = {}
    if _dev is not None:
        if env_opts.get("env", None) is None:
            env_opts["env"] = os.environ
        env_opts["env"]["PYTHONPATH"] = str(_dev)

    if port is None:
        port = find_unused_port()

    args = [
        "streamlit",
        "run",
        "--server.headless=True",
        "--theme.base=dark",
        "--theme.primaryColor=#E0735C",
        "--theme.font=sans-serif",
    ]
    if _watch_changes:
        args.extend([
            "--server.fileWatcherType=auto",
            "--client.toolbarMode=auto",
            "--global.disableWidgetStateDuplicationWarning=false",
        ])
    else:
        args.extend([
            "--server.fileWatcherType=none",
            "--client.toolbarMode=viewer",
            "--global.disableWidgetStateDuplicationWarning=true",
        ])

    if port is not None:
        args.append(f"--server.port={port}")
    if address is not None:
        args.append(f"--server.address={address}")

    args += [
        leaderboard_path,
        "--",
        "--database-prefix",
        session.connector.db.table_prefix,
    ]
    if (
        _is_snowflake_connector(session.connector)
        and not session.connector.password_known
    ):
        # If we don't know the password, this is problematic because we run the
        # dashboard in a separate process so we won't be able to recreate the
        # snowpark session in the child process. Thus, in this case we default
        # to using external browser authentication.
        # TODO: support other passwordless token authentication such as PAT.
        from trulens.connectors.snowflake.dao.sql_utils import (
            clean_up_snowflake_identifier,
        )

        connector = session.connector
        snowpark_session = connector.snowpark_session
        args_to_add = [
            ("--snowflake-account", snowpark_session.get_current_account()),
            ("--snowflake-user", snowpark_session.get_current_user()),
            ("--snowflake-role", snowpark_session.get_current_role()),
            ("--snowflake-database", snowpark_session.get_current_database()),
            ("--snowflake-schema", snowpark_session.get_current_schema()),
            ("--snowflake-warehouse", snowpark_session.get_current_warehouse()),
        ]
        for arg, val in args_to_add:
            if val:
                args += [arg, clean_up_snowflake_identifier(val)]
        args += ["--snowflake-authenticator", "externalbrowser"]
        if connector.use_account_event_table:
            args.append("--snowflake-use-account-event-table")
    else:
        args += [
            "--database-url",
            session.connector.db.engine.url.render_as_string(
                hide_password=False
            ),
        ]
    if sis_compatibility_mode:
        args += ["--sis-compatibility"]

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **env_opts,
    )

    started = threading.Event()
    tunnel_started = threading.Event()
    if notebook_utils.is_notebook():
        out_stdout, out_stderr = notebook_utils.setup_widget_stdout_stderr()
    else:
        out_stdout = None
        out_stderr = None

    if IN_COLAB:
        tunnel_proc = subprocess.Popen(
            ["npx", "localtunnel", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **env_opts,
        )

        def listen_to_tunnel(proc: subprocess.Popen, pipe, out, started):
            while proc.poll() is None:
                line = pipe.readline()
                if "url" in line:
                    started.set()
                    line = (
                        "Go to this url and submit the ip given here. " + line
                    )

                if out is not None:
                    out.append_stdout(line)

                else:
                    print(line)

        session._tunnel_listener_stdout = Thread(
            target=listen_to_tunnel,
            args=(tunnel_proc, tunnel_proc.stdout, out_stdout, tunnel_started),
        )
        session._tunnel_listener_stderr = Thread(
            target=listen_to_tunnel,
            args=(tunnel_proc, tunnel_proc.stderr, out_stderr, tunnel_started),
        )
        session._tunnel_listener_stdout.daemon = True
        session._tunnel_listener_stderr.daemon = True
        session._tunnel_listener_stdout.start()
        session._tunnel_listener_stderr.start()
        if not tunnel_started.wait(
            timeout=DASHBOARD_START_TIMEOUT
        ):  # This might not work on windows.
            raise RuntimeError("Tunnel failed to start in time. ")

    def listen_to_dashboard(proc: subprocess.Popen, pipe, out, started):
        while proc.poll() is None:
            line = pipe.readline()
            if IN_COLAB:
                if "External URL: " in line:
                    started.set()
                    line = line.replace(
                        "External URL: http://", "Submit this IP Address: "
                    )
                    line = line.replace(f":{port}", "")
                    if out is not None:
                        out.append_stdout(line)
                    else:
                        print(line)
                    session._dashboard_urls = (
                        line  # store the url when dashboard is started
                    )
            else:
                if "Local URL: " in line:
                    url = line.split(": ")[1]
                    url = url.rstrip()
                    print(f"Dashboard started at {url} .")
                    started.set()
                    session._dashboard_urls = (
                        line  # store the url when dashboard is started
                    )
                if out is not None:
                    out.append_stdout(line)
                else:
                    print(line)
        if out is not None:
            out.append_stdout("Dashboard closed.")
        else:
            print("Dashboard closed.")

    session._dashboard_listener_stdout = Thread(
        target=listen_to_dashboard,
        args=(proc, proc.stdout, out_stdout, started),
    )
    session._dashboard_listener_stderr = Thread(
        target=listen_to_dashboard,
        args=(proc, proc.stderr, out_stderr, started),
    )

    # Purposely block main process from ending and wait for dashboard.
    session._dashboard_listener_stdout.daemon = False
    session._dashboard_listener_stderr.daemon = False

    session._dashboard_listener_stdout.start()
    session._dashboard_listener_stderr.start()

    session._dashboard_proc = proc

    wait_period = DASHBOARD_START_TIMEOUT
    if IN_COLAB:
        # Need more time to setup 2 processes tunnel and dashboard
        wait_period = wait_period * 3

    # This might not work on windows.
    if not started.wait(timeout=wait_period):
        session._dashboard_proc = None
        raise RuntimeError(
            "Dashboard failed to start in time. "
            "Please inspect dashboard logs for additional information."
        )

    return proc


def stop_dashboard(
    session: Optional[core_session.TruSession] = None, force: bool = False
) -> None:
    """
    Stop existing dashboard(s) if running.

    Args:
        force: Also try to find any other dashboard processes not
            started in this notebook and shut them down too.

            **This option is not supported under windows.**

    Raises:
            RuntimeError: Dashboard is not running in the current process. Can be avoided with `force`.
    """
    session = session or core_session.TruSession()
    if session._dashboard_proc is None:
        if not force:
            raise RuntimeError(
                "Dashboard not running in this session. "
                "You may be able to shut other instances by setting the `force` flag."
            )

        else:
            if sys.platform.startswith("win"):
                raise RuntimeError(
                    "Force stop option is not supported on windows."
                )

            print("Force stopping dashboard ...")
            import os
            import pwd  # PROBLEM: does not exist on windows

            import psutil

            username = pwd.getpwuid(os.getuid())[0]
            for p in psutil.process_iter():
                try:
                    cmd = " ".join(p.cmdline())
                    if (
                        "streamlit" in cmd
                        and "Leaderboard.py" in cmd
                        and p.username() == username
                    ):
                        print(f"killing {p}")
                        p.kill()
                except Exception:
                    continue

    else:
        session._dashboard_proc.kill()
        session._dashboard_proc = None


def run_dashboard_sis(
    streamlit_name: str = "TRULENS_DASHBOARD",
    session: Optional[core_session.TruSession] = None,
    warehouse: Optional[str] = None,
    init_server_side_with_staged_packages: bool = False,
):
    with import_utils.OptionalImports(
        messages=import_utils.format_import_errors(
            "trulens-connectors-snowflake",
            purpose="running the TruLens dashboard in Streamlit in Snowflake",
        )
    ) as opt:
        import trulens.connectors.snowflake
    opt.assert_installed(trulens.connectors.snowflake)

    session = session or core_session.TruSession()

    if trulens.connectors.snowflake.SnowflakeConnector and isinstance(
        session.connector, trulens.connectors.snowflake.SnowflakeConnector
    ):
        return session.connector._set_up_sis_dashboard(
            streamlit_name,
            session.connector.snowpark_session,
            warehouse=warehouse,
            init_server_side_with_staged_packages=init_server_side_with_staged_packages,
        )
    else:
        raise ValueError(
            "This function is only supported with the SnowflakeConnector."
        )

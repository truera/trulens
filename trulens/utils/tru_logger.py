import logging

_logger = None


def log(level, msg, *args, **kwargs):
    get_logger().log(level, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    get_logger().debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    get_logger().error(msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
    get_logger().fatal(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    get_logger().info(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)


def get_logger():
    if _logger:
        return _logger
    configure()
    return _logger


def configure(lib_level=1, root_level=logging.WARNING):
    global _logger
    _logger = logging.getLogger(name="trulens")
    logging.basicConfig(format='%(levelname)s: %(message)s', level=root_level)
    _logger.setLevel(lib_level)

    _logger.info("lib level={lib_level}".format(lib_level=lib_level))
    _logger.info("root level={root_level}".format(root_level=root_level))

import os


def getenv(name):
    """Return the environment variable named `name`, or raise OSError if unset."""
    env = os.getenv(name)

    if env is None:
        raise OSError(f"Environment variable {name} is not set")

    return env

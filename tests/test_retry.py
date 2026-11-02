from app.agent.graph import is_transient_error


class Status(Exception):
    def __init__(self, code: int) -> None:
        self.status_code = code


def test_transient_errors_are_retried():
    assert is_transient_error(Status(429))
    assert is_transient_error(Status(529))
    assert is_transient_error(TimeoutError())


def test_client_errors_fail_fast():
    assert not is_transient_error(Status(400))
    assert not is_transient_error(Status(401))
    assert not is_transient_error(ValueError("temperature not supported"))

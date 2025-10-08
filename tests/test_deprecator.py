from typing import Any

import pytest

from pastas.decorators import PastasDeprecationWarning, deprecate_args_or_kwargs


def test_class_deprecation() -> None:
    # deprecate class in future version
    @PastasDeprecationWarning(remove_version="999.0.0", reason="Boo!")
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

    Deprecated(1)

    # class was deprecated in older version
    @PastasDeprecationWarning(remove_version="1.0.0", reason="Boo!")
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

    with pytest.raises(DeprecationWarning):
        Deprecated(1)


def test_classmethod_deprecation() -> None:
    # deprecate class in future version
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

        @PastasDeprecationWarning(remove_version="999.0.0", reason="Boo!")
        def foo(self, b: Any) -> Any:
            return self.a + b

    d = Deprecated(1)
    d.foo(2)  # warning

    # class was deprecated in older version
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

        @PastasDeprecationWarning(remove_version="1.0.0", reason="Boo!")
        def foo(self, b: Any) -> Any:
            return self.a + b

    with pytest.raises(DeprecationWarning):
        d = Deprecated(1)
        d.foo(2)  # raises error


def test_function_deprecation() -> None:
    # deprecate function in future version
    @PastasDeprecationWarning(remove_version="999.0.0", reason="Boo!")
    def foo(a: Any) -> None:
        print(a)

    foo(1)  # warning

    # function was deprecated in older version
    @PastasDeprecationWarning(remove_version="1.0.0", reason="Boo!")
    def foo(a: Any) -> None:
        print(a)

    with pytest.raises(DeprecationWarning):
        foo(1)  # raises error


def test_deprecate_args_or_kwargs() -> None:
    # log warning for future deprecation
    deprecate_args_or_kwargs("test", remove_version="999.0.0", reason="Boo!")

    # force error even for future deprecation
    with pytest.raises(DeprecationWarning):
        deprecate_args_or_kwargs(
            "test", remove_version="999.0.0", reason="Boo!", force_raise=True
        )

    # raise error for past deprecation
    with pytest.raises(DeprecationWarning):
        deprecate_args_or_kwargs("test", remove_version="1.0.0", reason="Boo!")

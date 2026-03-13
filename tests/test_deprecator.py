from typing import Any

import pytest

from pastas.decorators import PastasDeprecationWarning, deprecate_args_or_kwargs


def test_class_deprecation() -> None:
    # class will be removed in future version - should log warning
    @PastasDeprecationWarning(version="999.0.0", reason="Boo!")
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

    Deprecated(1)  # logs warning, continues execution

    # class was already removed (version <= current) - should raise AttributeError
    @PastasDeprecationWarning(version="0.1.0", reason="Boo!")
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

    with pytest.raises(AttributeError, match="module has no attribute"):
        Deprecated(1)


def test_classmethod_deprecation() -> None:
    # method will be removed in future version - should log warning
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

        @PastasDeprecationWarning(version="999.0.0", reason="Boo!")
        def foo(self, b: Any) -> Any:
            return self.a + b

    d = Deprecated(1)
    d.foo(2)  # logs warning, continues execution

    # method was already removed (version <= current) - should raise AttributeError
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

        @PastasDeprecationWarning(version="0.1.0", reason="Boo!")
        def foo(self, b: Any) -> Any:
            return self.a + b

    with pytest.raises(AttributeError, match="module has no attribute"):
        d = Deprecated(1)
        d.foo(2)  # raises error


def test_function_deprecation() -> None:
    # function will be removed in future version - should log warning
    @PastasDeprecationWarning(version="999.0.0", reason="Boo!")
    def foo(a: Any) -> None:
        print(a)

    foo(1)  # logs warning, continues execution

    # function was already removed (version <= current) - should raise AttributeError
    @PastasDeprecationWarning(version="0.1.0", reason="Boo!")
    def foo(a: Any) -> None:
        print(a)

    with pytest.raises(AttributeError, match="module has no attribute"):
        foo(1)  # raises error


def test_deprecate_args_or_kwargs() -> None:
    # log warning for future deprecation
    deprecate_args_or_kwargs("test", version="999.0.0", reason="Boo!")

    # raise TypeError when version has been reached
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        deprecate_args_or_kwargs("test", version="0.1.0", reason="Boo!")

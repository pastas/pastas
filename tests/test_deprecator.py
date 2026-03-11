from typing import Any

import pytest

from pastas.decorators import PastasDeprecationWarning, deprecate_args_or_kwargs


def test_class_deprecation() -> None:
    # class will be removed in future version - should log warning
    @PastasDeprecationWarning(
        deprecate_version="999.0.0", remove_version="1000.0.0", reason="Boo!"
    )
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

    Deprecated(1)  # logs warning, continues execution

    # class is currently deprecated (between deprecate and remove versions) - should raise DeprecationWarning
    @PastasDeprecationWarning(
        deprecate_version="1.0.0", remove_version="10.0.0", reason="Boo!"
    )
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

    with pytest.raises(DeprecationWarning, match="deprecated and is not available"):
        Deprecated(1)

    # class was already removed in past version - should raise AttributeError
    @PastasDeprecationWarning(
        deprecate_version="0.1.0", remove_version="1.0.0", reason="Boo!"
    )
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

        @PastasDeprecationWarning(
            deprecate_version="999.0.0", remove_version="1000.0.0", reason="Boo!"
        )
        def foo(self, b: Any) -> Any:
            return self.a + b

    d = Deprecated(1)
    d.foo(2)  # logs warning, continues execution

    # method is currently deprecated (between deprecate and remove versions) - should raise DeprecationWarning
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

        @PastasDeprecationWarning(
            deprecate_version="1.0.0", remove_version="10.0.0", reason="Boo!"
        )
        def foo(self, b: Any) -> Any:
            return self.a + b

    with pytest.raises(DeprecationWarning, match="deprecated and is not available"):
        d = Deprecated(1)
        d.foo(2)

    # method was already removed in past version - should raise AttributeError
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

        @PastasDeprecationWarning(
            deprecate_version="0.1.0", remove_version="1.0.0", reason="Boo!"
        )
        def foo(self, b: Any) -> Any:
            return self.a + b

    with pytest.raises(AttributeError, match="module has no attribute"):
        d = Deprecated(1)
        d.foo(2)  # raises error


def test_function_deprecation() -> None:
    # function will be removed in future version - should log warning
    @PastasDeprecationWarning(
        deprecate_version="999.0.0", remove_version="1000.0.0", reason="Boo!"
    )
    def foo(a: Any) -> None:
        print(a)

    foo(1)  # logs warning, continues execution

    # function is currently deprecated (between deprecate and remove versions) - should raise DeprecationWarning
    @PastasDeprecationWarning(
        deprecate_version="1.0.0", remove_version="10.0.0", reason="Boo!"
    )
    def foo(a: Any) -> None:
        print(a)

    with pytest.raises(DeprecationWarning, match="deprecated and is not available"):
        foo(1)

    # function was already removed in past version - should raise AttributeError
    @PastasDeprecationWarning(
        deprecate_version="0.1.0", remove_version="1.0.0", reason="Boo!"
    )
    def foo(a: Any) -> None:
        print(a)

    with pytest.raises(AttributeError, match="module has no attribute"):
        foo(1)  # raises error


def test_deprecate_args_or_kwargs() -> None:
    # log warning for future deprecation
    deprecate_args_or_kwargs(
        "test", deprecate_version="999.0.0", remove_version="1000.0.0", reason="Boo!"
    )

    # raise error when between deprecate and remove versions
    with pytest.raises(DeprecationWarning, match="is not available"):
        deprecate_args_or_kwargs(
            "test", deprecate_version="1.0.0", remove_version="10.0.0", reason="Boo!"
        )

    # raise TypeError when remove version has been reached
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        deprecate_args_or_kwargs(
            "test", deprecate_version="0.1.0", remove_version="1.0.0", reason="Boo!"
        )

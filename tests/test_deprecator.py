from typing import Any

import pytest

from pastas.decorators import deprecate_args_or_kwargs, deprecate_class_func_or_method

msg = "Boo!"


def test_class_deprecation() -> None:
    # class will be removed in future version - should show a FutureWarning
    @deprecate_class_func_or_method(version="999.0.0", reason=msg)
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

    with pytest.warns(FutureWarning, match=msg):
        Deprecated(1)  # logs warning, continues execution

    # class was already removed (version <= current) - should raise AttributeError
    @deprecate_class_func_or_method(version="0.1.0", reason=msg)
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

    with pytest.raises(AttributeError, match=msg):
        Deprecated(1)


def test_classmethod_deprecation() -> None:
    # method will be removed in future version - should show a FutureWarning
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

        @deprecate_class_func_or_method(version="999.0.0", reason=msg)
        def foo(self, b: Any) -> Any:
            return self.a + b

    d = Deprecated(1)
    with pytest.warns(FutureWarning, match=msg):
        d.foo(2)  # shows warning, continues execution

    # method was already removed (version <= current) - should raise AttributeError
    class Deprecated:
        def __init__(self, a: Any) -> None:
            self.a = a

        @deprecate_class_func_or_method(version="0.1.0", reason=msg)
        def foo(self, b: Any) -> Any:
            return self.a + b

    with pytest.raises(AttributeError, match=msg):
        d = Deprecated(1)
        d.foo(2)  # raises error


def test_function_deprecation() -> None:
    # function will be removed in future version - should show a FutureWarning
    @deprecate_class_func_or_method(version="999.0.0", reason=msg)
    def foo(a: Any) -> None:
        print(a)

    with pytest.warns(FutureWarning, match=msg):
        foo(1)  # shows warning, continues execution

    # function was already removed (version <= current) - should raise AttributeError
    @deprecate_class_func_or_method(version="0.1.0", reason=msg)
    def foo(a: Any) -> None:
        print(a)

    with pytest.raises(AttributeError, match=msg):
        foo(1)  # raises error


def test_deprecate_args_or_kwargs() -> None:
    # log warning for future deprecation
    deprecate_args_or_kwargs(
        "test", version="999.0.0", reason=msg
    )  # shows warning, continues execution

    # raise TypeError when version has been reached
    with pytest.raises(TypeError, match=msg):
        deprecate_args_or_kwargs("test", version="0.1.0", reason=msg)

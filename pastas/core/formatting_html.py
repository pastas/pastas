import uuid
import pastas as ps
from pandas import DataFrame
from functools import lru_cache
from html import escape
from importlib.resources import read_binary

STATIC_FILES = (
    ("pastas.static.html", "icons-svg-inline.html"),
    ("pastas.static.css", "style.css"),
)


@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed"""
    return [
        read_binary(package, resource).decode("utf-8")
        for package, resource in STATIC_FILES
    ]


def _obj_repr(obj, header_components, sections):
    """Return HTML repr of an xarray object.
    If CSS is not injected (untrusted notebook), fallback to the plain text repr.
    """
    header = f"<div class='xr-header'>{''.join(h for h in header_components)}</div>"
    sections = "".join(f"<li class='xr-section-item'>{s}</li>" for s in sections)

    icons_svg, css_style = _load_static_files()
    return (
        "<div>"
        f"{icons_svg}<style>{css_style}</style>"
        f"<pre class='xr-text-repr-fallback'>{escape(repr(obj))}</pre>"
        "<div class='xr-wrap' style='display:none'>"
        f"{header}"
        f"<ul class='xr-sections'>{sections}</ul>"
        "</div>"
        "</div>"
    )


def _icon(icon_name):
    # icon_name should be defined in xarray/static/html/icon-svg-inline.html
    return (
        "<svg class='icon xr-{0}'>"
        "<use xlink:href='#{0}'>"
        "</use>"
        "</svg>".format(icon_name)
    )


def array_section(ml, attrname, collapsed="checked"):
    # "unique" id to expand/collapse the section
    data_id = "section-" + str(uuid.uuid4())
    data_repr = short_data_repr_html(ml, attrname)
    # preview = escape(inline_variable_array_repr(variable, max_width=70))
    preview = attrname
    data_repr = short_data_repr_html(ml, attrname)
    data_icon = _icon("icon-file-text2")

    return (
        "<div class='xr-array-wrap'>"
        f"<input id='{data_id}' class='xr-array-in' type='checkbox' {collapsed}>"
        f"<label for='{data_id}' title='Show/hide data repr'>{data_icon}</label>"
        f"<div class='xr-array-preview xr-preview'><span>{preview}</span></div>"
        f"<div class='xr-array-data'>{data_repr}</div>"
        "</div>"
    )


def short_data_repr_html(ml, attrname):
    """Format "data" for object and Variable."""
    internal_data = getattr(ml, attrname, ml)
    if isinstance(internal_data, ps.TimeSeries):
        internal_data = internal_data.series.to_frame(name=internal_data.name)
    if isinstance(internal_data, dict):
        internal_data = DataFrame(internal_data, index=[attrname])
    if hasattr(internal_data, "_repr_html_"):
        return internal_data._repr_html_()

    text = "Niet Gelukt"
    return f"<pre>{text}</pre>"


def model_repr(ml):
    obj_type = f"pastas.{type(ml).__name__}"

    header_components = [
        f"<div class='xr-obj-type'>{obj_type}</div>",
        f"<div class='xr-array-name'>{ml.name}</div>",
    ]

    sections = [
        array_section(ml, "oseries"),
        array_section(ml, "settings", collapsed=""),
        array_section(ml, "parameters", collapsed=""),
    ]
    # return ml.oseries.series.to_frame()
    return _obj_repr(ml, header_components, sections)

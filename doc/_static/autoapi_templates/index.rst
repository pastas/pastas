API Reference
=============

This section contains the Documentation of the Application Programming Interface (API)
of Pastas. The information in this section is automatically created from the
documentation strings in original Python code. In the left-hand menu you will find the
different categories of the API documentation.

Submodules
----------

.. autoapisummary::

   {% for page in pages|rejectattr("is_top_level_object")|sort(attribute='id') %}
   {% if page.id.count('.') == 1 %}
   {{ page.id }}
   {% endif %}
   {% endfor %}

.. toctree::
   :hidden:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}


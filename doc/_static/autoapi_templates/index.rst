API Reference
=============

This section contains the Documentation of the Application Programming Interface (API) of Pastas. The information in this section is automatically created from the documentation strings in original Python code. In the left-hand menu you will find the different categories of the API documentation.

.. toctree::
   :titlesonly:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}


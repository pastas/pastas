==============================
Automatic Sphinx documentation
==============================

Follow these instructions when the documentation needs to be updated. Github stores the documentation in a special branch 
named "gh-pages". This branch is different from all other branches in the sense that it only contains the html files for
the documentation website (http://pastas.github.io/pasta/index.html).

| On your computer, a folder needs to be dedicated. The structure is as follows:
| ../Project/pasta-master/
| ../Project/pasta-docs/

The first folder is where the projects is maintained (the master branch) and the second folder is dedicated to the
documentation website. To set up your github for the pasta-doc folder, `see here 
<http://gisellezeno.com/tutorials/sphinx-for-python-documentation.html>`_.

Instructions for updating Sphinx documentation
---------------------------------------------
1. Open your terminal (MacOS) or command window (Windows)
2. Move to the doc folder in the pasta-master folder:
>>> cd ../path-to-pasta-master/doc

3. read pasta modules with sphinx-apidoc
  >>> sphinx-apidoc ../pasta -o .

4. Run sphinx auto documentation:
  >>> cd html

5. The html files have now been created in the html folder with the pasta-doc folder if Sphinxdoc was succesfull.
6. Move to the html folder with the pasta-docs folder:
  >>> cd
  >>> cd ../path-to-pasta-docs/html
  
7. Update the gh-pages branch:
  >>> git branch # Check if we are in the gh-pages branch
  >>> git add .
  >>> git commit -a -m "Update Docs"
  >>> git push origin gh-pages
  
Enjoy your the beautifully updates documentation!  

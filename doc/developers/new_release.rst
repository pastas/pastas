Make a Release
==============

To create a new release of Pastas follow the following steps:

1. Update version.py to the correct version ("X.X.X").
2. Commit these changes to the Dev-branch and sync.
3. Create a pull request to merge Dev into Master.
4. Check the documentation website if all is correct.
5. Merge commit the pull request. (Don't do squash and merge!).
6. Make a GitHub release from the master branch. A Pypi release will be
   created automatically and the Pastas version will receive a unique DOI at
   Zenodo.
7. Switch back to the Dev-branch and update the version.py file ("X.X.Xb").

.. warning::
    When making a release, do not use "Squash and merge", just do merge.

Dependency policy
-----------------

This project tries to follow `NEP29 <https://numpy
.org/neps/nep-0029-deprecation_policy.html>`_ and supports:

- All minor versions of Python released 42 months prior to the project, and
  at minimum the two latest minor versions.
- All minor versions of NumPy, Scipy, Matplotlib, and Pandas released in the
  24 months prior to the project, and at minimum the last three minor versions.

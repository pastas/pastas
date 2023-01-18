Make a Release
==============

To create a new release of Pastas follow the following steps:

1. Update version.py to the correct version.
2. Commit these changes to the Dev-branch and sync.
3. Create a pull request to merge Dev into Master.
4. Merge commit the pull request. (Don't do squash and merge!).
5. Make a GitHub release from the master branch. A Pypi release will be
   created automatically and the Pastas version will receive a unique DOI at
   Zenodo.
6. Switch back to the Dev-branch and update the version.py file ("X.X.Xb").

Dependency policy
-----------------

This project tries to follow `NEP29 <https://numpy
.org/neps/nep-0029-deprecation_policy.html>`_ and supports:

- All minor versions of Python released 42 months prior to the project, and
  at minimum the two latest minor versions.
- All minor versions of NumPy, Scipy, Matplotlib, and Pandas released in the
  24 months prior to the project, and at minimum the last three minor versions.

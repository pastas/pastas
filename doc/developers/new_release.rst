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

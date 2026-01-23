# Branching

As the project grows and community contributions become more frequent (thanks all!) we opted to adopt a branching model with the intention to make it easier and clearer for contributors to make pull requests.

We will mainly follow the model proposed at http://nvie.com/posts/a-successful-git-branching-model/ with some exceptions. After release 1.3.0 there will be two permanent branches: `main`and `develop`.

The `main` branch will be only be updated from `develop` for a new release. That is, each commit in `main` corresponds to a new release. In addition, there are temporary branches for bugfixes and features, which will mainly exist on developer’s forks. There may be public feature branches on aTrain in case a major feature is developed in collaboration.

After each release, a new branch `bugfixes_for_aTrain_1.x.x` will be opened. Any pull requests for bug fixes should use this base. The `bugfixes_for_aTrain_1.x.x` branch will be merged into `main` and `develop` for minor releases.

# Contributing the code

Developers create new branches on their own forks/repositories which are either named `feature_xx` for feature branches or `bugfixes_xx` for bugfix branches. Once they finished development, they can submit a pull request in order to merge their branch into `develop`. Before the pull requests, the branch should be rebased to the current state of develop such that the submitted changes are direct descendants from the current state of develop, and no merge conflicts exist.

If you want to contribute to a collaborative feature branch, the same steps apply as for contributing a full feature/bugfix to `develop`, only that the base will be that feature branch rather than `develop`.

aTrain maintainers may:

1. Reject a pull request - The feature may be deemed of no value to other aTrain users and not incorporated into the aTrain distribution. The discussion about this will be documented in the pull request comments. The developer may still provide his version to others, but there will be no support for this feature from aTrain maintainers. Feel of course free to contact us beforehand with any ideas.

2. Request modifications - aTrain maintainers may request changes to the submitted code. If the submitted pull request results in a merge conflict with the current state of `develop`, a common request will be to rebase the submitted branch to latest `develop`.

3. Accept the request - If the pull request is accepted, then the submitted branch will be merged into `develop`.

# Releases policy

Once `develop` reaches a state which justifies a new release, then there will be one additional house-keeping commit on `develop` (update version number, changelog, install and update instructions), and `develop` will be merged into `main`. The commit will be tagged as a release.

Release numbers conform to the following scheme: X.Y.Z with X, Y, Z natural numbers. Bugfixes and minor feature improvements (all of which do not feature any change to the database structure) result in an increase of Z. New features result in an increase of Y and setting Z to zero. Major new versions of aTrain with significant new features or code-rewriting will result in an increase of X and in setting Y and Z to zero. “alpha” and “beta” may be added to versions which are not stable yet but which nevertheless are proposed to the public for testing.

_Adapted for aTrain from [Orsee Development Instructions](https://github.com/orsee/orsee/) by @bgrainer_

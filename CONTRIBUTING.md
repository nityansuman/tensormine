# How to Contribute

## Issues

* Please tag your issue with `bug`, `feature request`, or `question` to help use effectively respond.
* Please include the versions of TensorFlow (run `pip list | grep tensor`)
* Please provide the command line you ran as well as the log output.

## How to become a contributor and submit your own code

### Contributor License Agreements

Follow either of the two links above to access the appropriate CLA and instructions for how to sign and return it. Once we receive it, we'll be able to accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed the CLA can be accepted into the main repository.

### Contributing code

If you have improvements to TensorHub, send us your pull requests! For those
just getting started, Github has a [howto](https://help.github.com/articles/using-pull-requests/).

TensorHub team members will be assigned to review your pull requests. Once the
pull requests are approved and pass continuous integration checks, a TensorHub
team member will apply `ready to pull` label to your change. This means we are
working on getting your pull request submitted to our internal repository. After
the change has been submitted internally, your pull request will be merged
automatically on GitHub.

If you want to contribute, start working through the TensorHub codebase,
navigate to the
[Github "issues" tab](https://github.com/nityansuman/tensorhub/issues) and start
looking through interesting issues. If you are not sure of where to start, then
start by trying one of the smaller/easier issues here i.e.
[issues with the "good first issue" label](https://github.com/nityansuman/tensorhub/labels/good%20first%20issue)
and then take a look at the
[issues with the "contributions welcome" label](https://github.com/nityansuman/tensorhub/labels/stat%3Acontributions%20welcome).
These are issues that we believe are particularly well suited for outside
contributions, often because we probably won't get to them right now. If you
decide to start on an issue, leave a comment so that other people know that
you're working on it. If you want to help out, but not alone, use the issue
comment thread to coordinate.


### Contribution guidelines and standards

Before sending your pull request for
[review](https://github.com/tensorflow/tensorflow/pulls),
make sure your changes are consistent with the guidelines and follow the
TensorFlow coding style.

### General guidelines and philosophy for contribution

*   Include unit tests when you contribute new features, as they help to a)
    prove that your code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
*   Bug fixes also generally require unit tests, because the presence of bugs
    usually indicates insufficient test coverage.
*   Keep API compatibility in mind when you change code in core TensorFlow,
    Reviewers of your pull request will comment on any API compatibility issues.
*   When you contribute a new feature to TensorHub, the maintenance burden is
    (by default) transferred to the TensorHub team. This means that benefit of
    the contribution must be compared against the cost of maintaining the
    feature.

### Python coding style

Changes to TensorHub Python code should conform to
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

Use `pylint` to check your Python changes. To install `pylint` and
retrieve TensorFlow's custom style definition:

```bash
pip install pylint
wget -O /tmp/pylintrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc
```

To check a file with `pylint`:

```bash
pylint --rcfile=/tmp/pylintrc myfile.py
```

### License

Include a license at the top of new files.
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-15 10:06:03 by Brian McFee <brian.mcfee@nyu.edu>
"""Helpful tools for deprecation"""

import warnings
from decorator import decorator

__all__ = ["moved", "deprecated"]


def moved(moved_from, version, version_removed):
    """This is a decorator which can be used to mark functions
    as moved/renamed.

    It will result in a warning being emitted when the function is used.
    """

    def __wrapper(func, *args, **kwargs):
        """Warn the user, and then proceed."""
        code = func.__code__
        warnings.warn_explicit(
            "{:s}\n\tThis function was moved to '{:s}.{:s}' in "
            "librosa version {:s}."
            "\n\tThis alias will be removed in librosa version "
            "{:s}.".format(
                moved_from, func.__module__, func.__name__, version, version_removed
            ),
            category=DeprecationWarning,
            filename=code.co_filename,
            lineno=code.co_firstlineno + 1,
        )
        return func(*args, **kwargs)

    return decorator(__wrapper)


def deprecated(version, version_removed):
    """This is a decorator which can be used to mark functions
    as deprecated.

    It will result in a warning being emitted when the function is used."""

    def __wrapper(func, *args, **kwargs):
        """Warn the user, and then proceed."""
        code = func.__code__
        warnings.warn_explicit(
            "{:s}.{:s}\n\tDeprecated as of librosa version {:s}."
            "\n\tIt will be removed in librosa version {:s}.".format(
                func.__module__, func.__name__, version, version_removed
            ),
            category=DeprecationWarning,
            filename=code.co_filename,
            lineno=code.co_firstlineno + 1,
        )
        return func(*args, **kwargs)

    return decorator(__wrapper)

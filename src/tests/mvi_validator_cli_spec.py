# -*- coding: utf-8 -*-

# =================================================================
# mvi-validator
#
# Copyright (c) 2022 Takahide Nogayama
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# =================================================================

from __future__ import unicode_literals, print_function, absolute_import

from uspec import description, context, it, execute_command
from hamcrest import assert_that, equal_to, greater_than

import mvi_validator

with description("mvi_validator command"):

    with context("--version option"):

        @it("prints version to stdout, and exit with status 0")
        def _(self):
            status: int
            stdout: bytes
            status, stdout, _ = execute_command(["mvi_validator", "--version"])
            assert_that(status, equal_to(0))
            assert_that(stdout.decode("UTF-8").strip(), equal_to(mvi_validator.__version__))

    for option in ["-h", "--help"]:

        with context(f"{option} option"):

            @it("prints help to stdout, and exit with status 0", option)
            def _(self, option):
                status: int
                stdout: bytes
                status, stdout, stderr = execute_command(["mvi_validator", option])
                assert_that(status, equal_to(0))
                assert_that(len(stdout), greater_than(0))


if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=2)

# -*- coding: utf-8 -*-

# =================================================================
# mvi-validator
#
# Copyright (c) 2022 Takahide Nogayama
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php
# =================================================================

from uspec import description, it
from hamcrest import assert_that, has_property

import mvi_validator

with description("mvi_validator"):

    @it("has __version__ field")
    def _(self):
        assert_that(mvi_validator, has_property("__version__"))


if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=2)

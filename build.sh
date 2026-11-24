#!/bin/bash
#
# @license
# Copyright 2026 FOAM Foundation and contributors
# SPDX-License-Identifier: Apache-2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

node foam3/tools/build.js "$@"

if [[ $? -eq 1 ]]; then
    tput bel;
fi

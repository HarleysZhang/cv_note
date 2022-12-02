#!/bin/bash

set -o nounset
set -o errexit
set -o pipefail

# Dir of script
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Variables
canvas="${DIR}/../node_modules/canvas"
a="'with_pango%': 'false'"
b="'with_pango%': 'true'"
gyp="node-gyp"

# Turn on pango support
bash -c "sed -i -e \"s/$a/$b/g\" ${canvas}/binding.gyp"

# Rebuild
bash -c "cd ${canvas} && ${gyp} rebuild"

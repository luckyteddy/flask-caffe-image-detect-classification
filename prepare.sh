#!/bin/bash

# Caffe evaluation with selective search.
#
# Copyright (c) 2016, Zdeněk Hřebíček
#
# This work is based on
# BVLC/Caffe web_demo example
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

readonly PROGDIR="$(readlink -m "$(dirname "${0}")")"
readonly CAFFE_URL="https://github.com/BVLC/caffe.git"
readonly BELLTAILP_URL="https://github.com/belltailjp/selective_search_py.git"
readonly ALPACADB_URL="https://github.com/AlpacaDB/selectivesearch.git"

declare -r -a  directoryStructutre=(
    'caffe'
    'selectivesearch'
    'selectivesearch/AlpacaDB'
    'selectivesearch/belltailjp'
)

printHelp()
{
    local -r progName="${0##*/}"

    cat << EOF

Script for preparation of folder structure and python code for Caffe evaluation
  with selective search.

Usage:

  ${progName} [-h|--help]

  ${progName} [--get-all]

  ${progName} [--get-selectivesearches]

Options:

  --get-all

    Clones all necessary git repositories into proper folders and gives
     instructions to make Caffe evaluation with selective search work.
     Including example model and auxiliary data.

  --get-selectivesearches

    Get only selectivesearches not caffe or example model and auxiliary data.

  -h, --help

    Print this help information and exit.

By default this script will just print help information like if it was invoked
with \`--help''.
EOF
}

get_caffe(){
    # Check if directory is empty
    if [ ! "$(ls -A ./caffe)" ]; then
        git clone "$CAFFE_URL" "./caffe"
    else
        direcry_contains_something "./caffe" "caffe"
    fi
}

get_selectivesearches(){
    # Check if directory is empty
    if [ ! "$(ls -A ./selectivesearch/belltailjp)" ]; then
        git clone "$BELLTAILP_URL" "./selectivesearch/belltailjp"
    else
        direcry_contains_something "./selectivesearch/belltailjp" "belltailjp"
    fi

    # Check if directory is empty
    if [ ! "$(ls -A ./selectivesearch/AlpacaDB)" ]; then
        git clone "$ALPACADB_URL" "./selectivesearch/AlpacaDB"
    else
        direcry_contains_something "./selectivesearch/AlpacaDB" "AlpacaDB"
    fi
}

direcry_contains_something(){
    directory="$1"; shift
    repo="$1"; shift
    cat << EOF

Something is allready in $directory folder hope it is realy selective search
  from $repo repo. If you are not sure that it contains what it should run:
  \`rm -rf $directory\` and than run this script again.
EOF
}

get_examples(){
    echo "Downloading example model"
    ./scripts/download_model_binary.py models/bvlc_reference_caffenet
    echo "Downloading auxiliary data "
    ./data/ilsvrc12/get_ilsvrc_aux.sh

}

prepare_directory_structure(){
    for dir in "${directoryStructutre[@]}"; do
        # check if dir exists
        if [[ ! -e "${dir}" ]]; then
            mkdir $dir
        fi
    done
}

print_info(){
    cat << _EOF_

################################# ALL DONE ##################################

Now you must manualy compile pycaffe; part or Caffe you can find how to do it
  here: http://caffe.berkeleyvision.org/installation.html.

Next thing you need is python2 and its depencencies which can be found here:
  ./web-app/dependencies.txt.

After that you are ready to go and run this app: \`python2 ./web-app/app.py\`,
  for more options run \`python2 ./web-app/app.py -h\`.

#############################################################################
_EOF_
}

main()
{
    local action='help'
    local exitCode=0

    while (( ${#} > 0 )); do
        arg="${1}"; shift
        case "${arg}" in
          '-h'|'--help'|'-help')
            action='help'
            break
            ;;
          '--get-all')
            action='get-everything'
            break
            ;;
           '--get-selectivesearches')
             action='get-selectivesearches'
             break
             ;;
          *)
            action='help'
            printf "\`%s': Unknown option.\n\n" "${arg}" >&2
            exitCode=1
            ;;
        esac
    done

    (
        # Change to scriptdir to be able to create right directory structure
        cd "${PROGDIR}"
        case "${action}" in
          'help')
            printHelp
            ;;
          'get-everything')
            prepare_directory_structure
            get_caffe
            get_selectivesearches
            (
                cd ./caffe
                get_examples
            )
            print_info
            ;;
          'get-selectivesearches')
            prepare_directory_structure
            get_selectivesearches
            print_info
            ;;
          *)
            echo "Unhandled action '${action}'" >&2
            exit 127
            ;;
        esac
    )

    # Propagate exit from previous subshell.
    [ $? -ne 0 ] && exit $?

    exit ${exitCode}
}

main "$@"

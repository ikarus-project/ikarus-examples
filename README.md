<!--
SPDX-FileCopyrightText: 2021-2025 The Ikarus Developers ikarus@ibb.uni-stuttgart.de
SPDX-License-Identifier: LGPL-3.0-or-later
-->

[![Debian](https://github.com/ikarus-project/ikarus-examples/actions/workflows/debian.yml/badge.svg)](https://github.com/ikarus-project/ikarus-examples/actions/workflows/debian.yml)
[![CodeStyle](https://github.com/ikarus-project/ikarus-examples/actions/workflows/style.yml/badge.svg)](https://github.com/ikarus-project/ikarus-examples/actions/workflows/style.yml)
[![codespell](https://github.com/ikarus-project/ikarus-examples/actions/workflows/codespell.yml/badge.svg)](https://github.com/ikarus-project/ikarus-examples/actions/workflows/codespell.yml)
# Ikarus
## Examples

This repository tries to provide various problems solved with [Ikarus](https://ikarus-project.github.io/), thereby serving as a sandbox of examples used to understand the working of the tool itself.
The examples are motivated within the finite element framework.
In order to only work with the examples of Ikarus, the ikarus-examples repository is to be downloaded.
The simplest way to use ikarus is pulling the docker image by using the following command:
```sh
docker pull ikarusproject/ikarus:latest
```
More details on the installation can be found in the [documentation](https://ikarus-project.github.io/download/).

The examples can be executed using a software like Clion or directly from a command line as shown below.

```sh
docker container run -it --entrypoint /bin/bash  ikarusproject/ikarus:latest 
git clone https://github.com/ikarus-project/ikarus-examples.git &&
cd ikarus-examples &&
mkdir build &&
cd build &&
cmake ../ -DCMAKE_BUILD_TYPE=Release &&
cmake --build . --parallel 2 --target <filenameWithoutExtension>
cd src
./<filenameWithoutExtension>
```

The corresponding documentation for existing examples resides at the [link](https://ikarus-project.github.io/dev/02_examples/).

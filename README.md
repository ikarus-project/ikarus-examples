<!--
SPDX-FileCopyrightText: 2022 The Ikarus Developers mueller@ibb.uni-stuttgart.de
SPDX-License-Identifier: LGPL-2.1-or-later
-->

[![Debian](https://github.com/IkarusRepo/IkarusExamples/actions/workflows/debian.yml/badge.svg)](https://github.com/IkarusRepo/IkarusExamples/actions/workflows/debian.yml)
[![CodeStyle](https://github.com/IkarusRepo/IkarusExamples/actions/workflows/style.yml/badge.svg)](https://github.com/IkarusRepo/IkarusExamples/actions/workflows/style.yml)
[![codespell](https://github.com/IkarusRepo/IkarusExamples/actions/workflows/codespell.yml/badge.svg)](https://github.com/IkarusRepo/IkarusExamples/actions/workflows/codespell.yml)
# Ikarus
## Examples

This repository tries to provide various problems solved with [Ikarus](https://ikarusrepo.github.io/), thereby serving as a sandbox of examples used to understand the working of the tool itself.
The examples are motivated within the finite element framework.
In order to only work with the examples of Ikarus, the IkarusExamples repository is to be downloaded and Ikarus could be pulled by using the following command in PowerShell:
```sh
docker pull rath3t/ikarus:latest
```
More details on the installation can be found in the [documentation](https://ikarusrepo.github.io/download/).

The examples can be executed using a software like Clion or directly from a command line as shown below.

```sh
docker container run -it --entrypoint /bin/bash  rath3t/ikarus:latest 
git clone https://github.com/IkarusRepo/IkarusExamples.git &&
cd IkarusExamples &&
mkdir build &&
cd build &&
cmake ../ -DCMAKE_BUILD_TYPE=Release &&
cmake --build . --parallel 2 --target iksXXX
./src/iksXXX
```
`iksXXX` should be replaced by the desired executable file in the above-mentioned set of commands.

A documentation on existing examples resides at https://ikarusrepo.github.io/examples/.

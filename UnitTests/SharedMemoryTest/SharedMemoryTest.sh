#!/bin/sh
echo -ne '\033c\033]0;SharedMemoryTest\a'
base_path="$(dirname "$(realpath "$0")")"
"$base_path/SharedMemoryTest.x86_64" "$@"

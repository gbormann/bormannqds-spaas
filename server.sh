#!/usr/bin/bash

if [ ! -f "src/py/spaas.py" ];
then
  echo "Please run from the project directory!"
  exit 1
fi

python3 src/py/spaas.py

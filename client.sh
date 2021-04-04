#!/usr/bin/bash

if [ ! -f "examples/payloads/" ];
then
  echo "Please run from the project directory!"
  exit 1
fi

if [ ! -f "$1" ];
then
	echo "$1 not found!"
	echo "Try f.i. one of examples/payloads/payload1.json, examples/payloads/payload2.json or examples/payloads/payload3.json..."
	exit 1
fi

curl -X POST -H "Accept: application/json" -H "Content-Type: application/json" -T $1 http://localhost:8888/v1/productionplan


# SPaaS Production Planner API

## Intro

My attempt at Engie's GEM-SPaaS PowerPlant Coding Challenge :-)
(See: https://github.com/gem-spaas/powerplant-coding-challenge)

## Prerequisites

This is a Python3 (v3.7) application. (python3 is expected to be on the path irrespective of run-time environment.)

### Planner implementation

The production planner implementation only requires standard modules.

### API controller implementation and server

The API controller is a Flask+Flask-RESTful application and the demo relies on
the  demo/test server that comes with Flask.

Dependencies:
* Flask
* Flask-RESTful

pip3'ing these two modules resolves all the dependencies that are needed to run the demo.
(The requirements.txt specifies the full dependency resolution.)

## Installation

It's sufficient to just clone the repository from GitHub. The demo can be ran directly
from the project directory.

The application consists of two files in src/py/ :
* powerplanner.py: the module that contains the ProductionPlanner class used by the API controller.
* spaas.py: the module that specifies the Flask-RESTful resource controller, bound to a Flask Api object.

## Running
Two simple driver scripts are provided that have to be ran from the project directory as it uses relative paths:
* server.sh: this starts a simple debug server that listens on port 8888
* client.sh: this uses curl to POST a JSON payload taken from a provided input file (first argument; additional 
args are ignored) on the /v1/productionplan endpoint to trigger a plan generation.

The application logs to stdout through a logger. Setting the log level to DEBUG in src/py/powerplanner.py sheds some
light on the inner workings of the production planner.

A third script, test_planner.sh, is provided to run a quick 'unit' test of ProductionPlanner. The test cycles
through all the example payloads in the examples/payloads/ directory and writes corresponding solutions in the
examples/plans/ directory (in files named responseN.json, N=1 for input payload1.json etc).
(It does not rely on a test framework; it's just a simple driver function for the class.)

NOTE: The Flask debug server should never run in Production. If you insist on going against this security advice,
at least hide it behind a Proxy/Reverse Proxy combo (e.g. using the Apache Httpd mod_proxy module).

## Future improvements

* Add WebSocket to broadcast the response to the latest request.
* Running from a Docker container (already partially setup but not yet ready for re-imaging).
(httpd proxying is already tested)
* Add input validation on the JSON request payload (this was my first Flask application).

import sys
import logging

from powerplanner import UnderpoweredError, PowerGapError, ProductionPlanner

from flask import Flask, request
from flask_restful import Resource, Api

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout
)


class Spaas(Resource):
    def __init__(self):
        self.m_log = logging.getLogger("Spaas")

    def post(self):
        payload = request.json
        self.m_log.info(payload)
        planner = ProductionPlanner(payload)
        try:
            planner.validate()
            plan_tp = planner.generate_plan()
            self.m_log.info("")
            self.m_log.info(f"Plan at total cost of {plan_tp[0]}; load fully committed.")
            self.m_log.info("-----------------------------------------------------")
            for pp in plan_tp[3]:
                self.m_log.info(f"{pp.m_name}: type={pp.m_type} unit cost={pp.unit_cost()} load={pp.load()}")
            self.m_log.info("-----------------------------------------------------")
        except PowerGapError as pge:
            self.m_log.error(f"Power gap! {pge}")
            return {'message': f"Power gap! {pge}"}, 404
        except UnderpoweredError as upe:
            self.m_log.error(f"Underpowered! {upe}")
            return {'message': f"Underpowered! {upe}"}, 400
        else:
            result = [
                tp[2] for tp in sorted(
                    [(pp.unit_cost(), pp.m_name, {'name': pp.m_name, 'p': pp.load()}) for pp in planner.m_pps]
                )
            ]
            self.m_log.info(result)
            return result


def main():
    app = Flask(__name__.split('.')[0])

    api = Api(app, "/v1", "application/json")
    api.add_resource(Spaas, "/productionplan")

#    app.run(debug=True)
    app.run(port=8888, debug=True)


if __name__ == "__main__":
    main()


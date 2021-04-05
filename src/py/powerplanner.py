import sys
import enum
import heapq as heap
import operator as op
import functools as fn
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout
)


class PpType(enum.Enum):
    """Enum representing power plant types"""
    GASFIRED = 'gasfired'
    TURBOJET = 'turbojet'
    WINDTURBINE = 'windturbine'


class Error(Exception):
    """Base class for errors in this module."""
    pass


class BoundaryConditionViolationError(Error):
    """Errors raised when the expected demand load cannot be met by the provided list of power generation assets."""
    pass


class UnderpoweredError(BoundaryConditionViolationError):
    """The power generation capacity of the given list of assets cannot meet the expected demand load.

    Atrributes:
        message -- explanatory details
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class PowerGapError(BoundaryConditionViolationError):
    """
    The power generation configuration space of the given list of assets has a power gap. No solution can be found
    that meets the expected demand load while satisfying the minimum power requirements of certain assets.

    Atrributes:
        message -- explanatory details
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class Fuel(object):
    """
    Class representing a fuel consumed by a particular power plant type.
    (Dim. an.: Power = Energy / Time but demand load is estimated in 1 hour intervals so 'power' and 'energy' can be
     used interchangeably when energy is expressed in units of MWh.)

    Attributes:
        price -- Unit price in € of 1 MWh of energy released in 'burning' the fuel
        co2_premium -- The certificate premium in € paid per tonne of released carbon dioxide
        kappa_co2 -- Tonnes of CO2 release per MWh of electric energy generated (given and fixed)
        phi -- Fuel supply flux (average local specific wind flux; idealised to 1. for other fuel types)
    """
    def __init__(self, price, co2_premium, kappa_co2, phi):
        self.m_price = price  # € / MWh(released)
        self.m_co2_premium = co2_premium  # € / tonne
        self.m_kappa_co2 = kappa_co2  # tonne / MWh(generated), this glosses over the effects of inefficiency on CO2 emissions
        self.m_phi = phi  # supply flux

    # factory class method
    @classmethod
    def create_fuels(cls, fuels_config):
        """Factory method to create all fuel objects from the 'fuels' section of the JSON input payload"""
        fuels = {}
        co2_premium = fuels_config['co2(euro/ton)']
        fuels[PpType.GASFIRED] = cls(fuels_config['gas(euro/MWh)'], co2_premium, .3, 1.)
        fuels[PpType.TURBOJET] = cls(fuels_config['kerosine(euro/MWh)'], co2_premium, .3, 1.)
        # Apparently there is no local weather in Electroland. So wind flux phi is a constant, not a field.
        # And no operating costs!!
        fuels[PpType.WINDTURBINE] = cls(0., 0., 0., fuels_config['wind(%)'] / 100.)
        return fuels

    def price(self):
        """Returns the current fuel unit price (might be supplied by a TOB market data feed hence the method)."""
        return self.m_price

    def total_cost(self, p_gen, eta):
        """Calculates total cost for a given amount of energy p_gen at given plant efficiency eta"""
        return (self.m_price / eta + self.m_kappa_co2 * self.m_co2_premium) * p_gen

    def phi(self):
        """Returns the current supply flux (might be fed from telemetry hence the method)"""
        return self.m_phi


class PowerPlant(object):
    """
    Class representing a power plant of a given type.
    
    Attributes:
        name -- symbolic name to identify a specific plant
        type -- the type of plant (codified as a value of type PpType)
        eta -- plant efficiency (MWh of energy generated per MWh of energy released from 'burning' fuel)
        p_min -- Minimum energy generated when the plant is switched on
        p_max -- Maximum electric energy the plant can generate; for wind-turbines this is modulated by wind flux
        fuel -- Fuel object representing the fuel 'burned' by the plant
    """
    def __init__(self, name, type, eta, p_min, p_max, fuel):
        self.m_name = name
        self.m_type = type
        self.m_eta = eta
        self.m_p_min = p_min  # in MWh
        self.m_p_max = p_max  # in MWh
        #        self.m_DelT = 1. # in hr
        self.m_fuel = fuel
        self.m_load = 0.

    # factory class method
    @classmethod
    def create_power_plants(cls, pp_configs, fuels):
        """Factory method to create all power plant objects from the 'power plants' section of the JSON input payload"""
        return [cls(c['name'], PpType(c['type']), c['efficiency'], float(c['pmin']), float(c['pmax']),
                    fuels[PpType(c['type'])])
                for c in pp_configs]

    def load(self):
        """Returns the current load, rounded to the nearest 100kWh"""
        return round(10. * self.m_load) / 10.

    def reset_load(self):
        """Resets the current load to 0"""
        self.m_load = 0.

    def max_load(self):
        """Returns the maximum load the plant can sustain"""
        self.m_load = self.p_max()

    def is_underloaded(self):
        """Indicates whether the plant is underloaded relative to the minimum energy it generates"""
        return self.m_load < self.m_p_min

    def add_load(self, load):
        """
        Add a given amount of load and return the effective load added,
        clipping the load at the maximum amount of electric energy the plant can generate
        """
        dload = min(self.p_max() - self.m_load, load)
        self.m_load += dload
        return dload

    def repartition_load(self, load):
        """
        Reduce the current load with a given amount and return the effective amount of reduction,
        clamping the load at the minimum amount of energy the plant will generate
        """
        dload = max(0., min(self.m_load - self.m_p_min, load))
        self.m_load -= dload
        return dload

    def p_max(self):
        """Returns the effective maximum generated energy given the fuel flux"""
        return round(10. * self.m_fuel.phi() * self.m_p_max) / 10.

    def p_gen(self, load):
        """
        Returns the effectively generated energy for a given amount of demand load,
        taking into account the power profile of the plant
        """
        return min(max(self.m_p_min, load), self.p_max())  # technically DelT * p_{min|max}

    def unit_cost(self):
        """Returns the unit cost of 1 MWh of generated electric energy, given the plant's energy"""
        return self.m_fuel.total_cost(1., self.m_eta)

    def load_cost(self, load):
        """
        Returns the total cost to produce the energy necessary to sustain the given demand load,
        _ignoring_ the power profile of the plant
        """
        return load * self.unit_cost()

    def cur_load_cost(self):
        """
        Returns the total cost to produce the energy necessary to sustain the current load,
        _taking into account_ the power profile of the plant provided the load is set through the class methods
        """
        return self.m_load * self.unit_cost()

    def budget_req(self, load):
        """
        Returns the required budget to sustain the part of the given demand load that is compatible with
        the power profile of the plant
        """
        return self.m_fuel.total_cost(self.p_gen(load), self.m_eta)

    def budget_reserved(self):
        """
        Returns the budget that would have to be reserved to generate the energy needed to sustain the demand load
        currently committed to the plant
        """
        return self.budget_req(self.m_load)


class ProductionPlanner(object):
    """
    Class that implements the production planner.

    The planner uses a priority queue of power plants sorted in merit order to perform a kind of algorithmic gradient
    descent (with defined jumps to take into account the minimum power requirements). This is possible because the merit
    order (with a suitable tie-breaker) imposes a total ordering on the set of power plants.

    ALGO DETAILS

    Given a demand load it peels off power plants in merit order from the queue and recursively rebalances
    the assigned demand load commitments until the minimum power requirement is met for the lowest merit (i.e. 'last')
    power plant in the proposed solution that satisfies the total demand load.

    It does this by comparing

    [the total cost of the amended solution obtained by repartitioning a sufficient fraction
    of the committed load from higher merit plants to satisfy the minimum power requirement]

    with

    [the total cost of the amended solution obtained by recommitting the load from the last underloaded lower merit
    power plant to an even lower merit power plant (i.e. skipping power plants with equal minimum power requirements).]

    Recommitting load from higher merit plants to lower merit plants to lower cost might seem counter-intuitive but
    there is a tipping point below which committing a small amount of load to a lower merit plant is cheaper than the
    opportunity cost of taking away a large amount of committed load from the higher merit plants to satisfy the
    minimum power requirements of the underloaded last plant of the current iteration.

    Above the tipping point it is cheaper to repartition already committed load from the higher merit plants.

    ERROR CONDITIONS

    An obvious error condition is when the total electric energy that can be generated by the given set of assets is
    not sufficient to sustain the demand load.

    A not so obvious second error condition is when the combination of
    minimum power requirements is such that there is no available capacity to sustain particular intervals of power
    despite sufficient maximum power generation capacity (typically under conditions of low wind flux).

    Attributes:
        pl -- the JSON request payload
    """
    def __init__(self, pl):
        self.m_log = logging.getLogger("ProductionPlanner")
        # filters and lambdas: list(filter(lambda pp: pp['type'] == 'windturbine', pl['powerplants']))
        fuels = Fuel.create_fuels(pl['fuels'])

        self.m_pps = PowerPlant.create_power_plants(pl['powerplants'], fuels)

        self.m_exp_load = pl['load']
        #        self.m_exp_load = 165
        #        self.m_exp_load = 185
        #        self.m_exp_load = 623
        #        self.m_exp_load = 1031.6
        #        self.m_exp_load = 1257.6
        #        self.m_exp_load = 36 # Catch22! with wind(%)<=10

        self.m_pp_mo_q = [(self.m_pps[i].unit_cost(), self.m_pps[i].p_max(), i, self.m_pps[i]) for i in
                          range(0, len(self.m_pps))]
        heap.heapify(self.m_pp_mo_q)

    def validate(self):
        load_min = fn.reduce(op.add,
                             [pp.p_max() for pp in list(filter(lambda pp: self.m_exp_load > pp.m_p_min, self.m_pps))],
                             0.)
        load_max = fn.reduce(op.add, [pp.p_max() for pp in self.m_pps])
        self.m_log.info(f"E[load]={self.m_exp_load} vs [min(load)={load_min}, max(load)={load_max}]")
        if self.m_exp_load > load_min:
            self.m_log.info("Catch22!")
        if load_max < self.m_exp_load:
            self.m_log.error("Once upon a time in Texas...")
            raise UnderpoweredError(
                f"Expected demand load {self.m_exp_load} cannot be met by power generation capacity of {load_max}")

    def display_pp(self, prefix, pp):
        """(Used for debugging)"""
        self.m_log.debug(f"{prefix}{pp.m_name} {pp.m_type}")

    def display_merit_order(self):  # to log or to dashboard
        """(Used for debugging)"""
        for pp in heap.nsmallest(len(self.m_pps), self.m_pp_mo_q):
            self.m_log.debug(
                f"{pp[3].m_name}: type={pp[3].m_type} unit cost={pp[0]} p_min={pp[3].m_p_min} p_max={pp[1]} load={pp[3].load()}"
            )

    def try_recommit(self, load_to_recommit, plan_tp):
        """Try ammending the solution for the given plan tuple by recommitting the given load to lower merit plants"""
        initial_cost = plan_tp[0]
        prev_pp = plan_tp[2]

        res = self.__allocate_load(load_to_recommit, plan_tp, True, True)
        cost_recommitting = initial_cost - prev_pp.cur_load_cost() + res[0]

        self.m_log.info(f"Cost of recommitting: {cost_recommitting}; uncovered load: {res[1]}")
        for pp_tp in self.m_pp_mo_q: pp_tp[3].reset_load()

        return cost_recommitting, res[1]

    def try_repartitioning(self, load_to_repartition, plan_tp):
        """
        Try ammending the solution for the given plan tuple by repartitioning the given load from higher merit plants
        """
        initial_cost = plan_tp[0]
        prev_pp = plan_tp[2]
        plan = plan_tp[3]

        res = self.__repartition_load(load_to_repartition, plan)
        cost_repartitioning = initial_cost + prev_pp.budget_reserved() - prev_pp.cur_load_cost() - res[0]

        self.m_log.info(f"Cost of repartitioning: {cost_repartitioning}; uncovered load: {res[1]}")
        for i in range(2, len(plan) + 1): self.m_pps[plan[-i]].max_load()

        return cost_repartitioning, res[1]

    def rebalance(self, plan_tp):
        """
        Recursively improving a solution by bifurcating on the cost for resolving the load currently assigned to the
        last underloaded power plant in the latest solution iteration
        """
        uncovered_load = plan_tp[1]
        prev_pp = plan_tp[2]

        if prev_pp is None or not prev_pp.is_underloaded():
            if uncovered_load > 0.:
                raise PowerGapError(f"{uncovered_load} of demand load cannot be covered by remaining capacity")
            return plan_tp

        cost = plan_tp[0]
        plan = plan_tp[3]

        self.m_log.info(f"Plan budget required is {cost} for {uncovered_load} uncovered")
        self.m_log.info(f"\t\t({prev_pp.m_name} at {prev_pp.load()} underloaded? {prev_pp.is_underloaded()})")
        self.display_pp("\tprev: ", prev_pp)
        committed_load = self.m_exp_load - prev_pp.load()
        load_to_repartition = round(10 * (prev_pp.m_p_min - prev_pp.load())) / 10
        load_to_recommit = prev_pp.load()
        self.m_log.info(
            f"p_min({prev_pp.m_p_min}): load to recommit {load_to_recommit} vs load to repartition {load_to_repartition}")

        recommit_result = self.try_recommit(load_to_recommit, plan_tp)
        repartition_result = self.try_repartitioning(load_to_repartition, plan_tp) if committed_load > 0 else (
        1.e9, load_to_repartition)
        is_recommit_cheaper = recommit_result[0] < repartition_result[0]

        uncovered_load = 0.
        if committed_load > 0 and (not is_recommit_cheaper or recommit_result[1] > 0.):
            self.m_log.info("Repartitioning...")
            res = self.__repartition_load(load_to_repartition, plan)
            cost += prev_pp.budget_reserved() - prev_pp.cur_load_cost() - res[0]

            prev_pp.m_load = prev_pp.m_p_min
            tmp_pp = heap.heappop(self.m_pp_mo_q)[3]
            self.display_pp("\ttop: ", tmp_pp)
            uncovered_load = repartition_result[1]
        elif is_recommit_cheaper or repartition_result[1] > 0.:
            self.m_log.info("Recommitting...")
            plan.pop()
            res = self.__allocate_load(load_to_recommit, plan_tp, True, False)
            cost += res[0] - prev_pp.cur_load_cost()
            uncovered_load = recommit_result[1]

            prev_pp.reset_load()
            prev_pp = res[2]
        # else: boter, noch vis

        self.m_log.info(f"Adjusted budget: {cost}; uncovered load: {uncovered_load}")

        return self.rebalance((cost, uncovered_load, prev_pp, plan))

    def generate_plan(self):
        """Driver method triggering the rebalancing dance on an Ansatz solution"""
        plan_tp = self.rebalance(self.__initial_commit())
        return plan_tp[0], plan_tp[1], plan_tp[2], [self.m_pps[i] for i in plan_tp[3]]

    def __allocate_load(self, load, plan_tp, is_recommit, is_trial):
        prev_pp = plan_tp[2]
        top_pp = None
        cost = 0.
        try:
            pps = self.m_pp_mo_q.copy() if is_recommit else self.m_pp_mo_q
            while load > 0.:
                pp = pps[0][3]
                self.display_pp("\tscan: ", pp)
                if (not is_recommit) or pp.unit_cost() > prev_pp.unit_cost() and pp.m_p_min <= prev_pp.m_p_min:
                    load -= pp.add_load(load)
                    cost += pp.cur_load_cost()
                    if not is_trial and pp.p_max() > 0.:
                        plan_tp[3].append(pps[0][2])
                top_pp = heap.heappop(pps)[3]
                self.display_pp("\ttop: ", top_pp)
        except IndexError:
            pass

        return cost, load, top_pp

    def __repartition_load(self, load, plan):
        savings_repartition = 0.
        for i in range(2, len(plan) + 1):
            prev_cost = self.m_pps[plan[-i]].cur_load_cost()
            load -= self.m_pps[plan[-i]].repartition_load(load)
            savings_repartition += prev_cost - self.m_pps[plan[-i]].cur_load_cost()
            if load <= 0.:
                break

        return savings_repartition, load

    def __initial_commit(self):
        plan_tp = (0., self.m_exp_load, None, [])
        res = self.__allocate_load(self.m_exp_load, plan_tp, False, False)
        return res[0], res[1], res[2], plan_tp[3]


def test_plan(payload_uri, result_uri):
    """Simple planner test driver. To be replaced with a proper unit test."""
    log = logging.getLogger(__name__)
    log.info(f"--- {payload_uri} ---")
    pl_file = open(payload_uri, "r")
    planner = ProductionPlanner(json.load(pl_file))
    pl_file.close()
    try:
        planner.validate()
        plan_tp = planner.generate_plan()
        log.info("")
        log.info(f"Plan at total cost of {plan_tp[0]}; load fully committed.")
        log.info("-_-----------------------------------------------------------_-")
        for pp in plan_tp[3]:
            log.info(f"{pp.m_name}: type={pp.m_type} unit cost={pp.unit_cost()} load={pp.load()}")
        log.info("-_-----------------------------------------------------------_-")
    except PowerGapError as pge:
        log.error(f"Power gap! {pge}")
    except UnderpoweredError as upe:
        log.error(f"Underpowered! {upe}")
    else:
        res_json = json.dumps(
            [
                tp[2] for tp in sorted(
                    [(pp.unit_cost(), pp.m_name, {"name": pp.m_name, "p": pp.load()}) for pp in planner.m_pps]
                )
            ],
            indent="\t"
        )
        log.info(res_json)
        resp_file = open(result_uri, "w")
        json.dump(
            [
                tp[2] for tp in sorted(
                    [(pp.unit_cost(), pp.m_name, {"name": pp.m_name, "p": pp.load()}) for pp in planner.m_pps]
                )
            ],
            resp_file,
            indent="\t"
        )
        resp_file.close()


def main():
    """Test run entrypoint"""
    test_plan("examples/payloads/payload1.json", "examples/plans/response1.json")
    test_plan("examples/payloads/payload2.json", "examples/plans/response2.json")
    test_plan("examples/payloads/payload3.json", "examples/plans/response3.json")


# Enable a simple test run
if __name__ == "__main__":
    main()

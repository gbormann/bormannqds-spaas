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
    GASFIRED = 'gasfired'
    TURBOJET = 'turbojet'
    WINDTURBINE = 'windturbine'

class BndType(enum.Enum):
    FREE = 0		# cost break-even point between stealing from higher merit PP and recommitting to lower merit PP 
    MAX_STEAL = 1	# sub-optimal cost at max stealable power boundary
    MAX_RECOMMIT = 2	# sub-optimal cost at max recommittable power boundary

class Fuel(object):
    def __init__(self, price, co2_premium, kappa_co2, phi):
        self.m_price = price # € / MWh(released)
        self.m_co2_premium = co2_premium # € / ton
        self.m_kappa_co2 = kappa_co2 # ton / MWh(generated), this glosses over the effects of inefficiency on CO2 emissions
        self.m_phi = phi # supply flux

    # factory class method
    @classmethod
    def createFuels(cls, fuels_config):
        fuels = {}
        co2_premium = fuels_config['co2(euro/ton)']
        fuels[PpType.GASFIRED] = cls(fuels_config['gas(euro/MWh)'], co2_premium, .3, 1.)
        fuels[PpType.TURBOJET] = cls(fuels_config['kerosine(euro/MWh)'], co2_premium, .3, 1.)
        # Apparently there is no local weather in Electroland. So wind flux phi is a constant, not a field.
        fuels[PpType.WINDTURBINE] = cls(0., 0., 0., fuels_config['wind(%)'] / 100.)
        return fuels

    def price(self):
        return self.m_price

    def total_cost(self, p_gen, eta):
        return (self.m_price / eta + self.m_kappa_co2 * self.m_co2_premium) * p_gen

    def phi(self):
        return self.m_phi

class PowerPlant(object):
    def __init__(self, name, type, eta, p_min, p_max, fuel):
        self.m_name = name
        self.m_type = type
        self.m_eta = eta
        self.m_p_min = p_min # in MW
        self.m_p_max = p_max # in MW
#        self.m_DelT = 1. # in hr
        self.m_fuel = fuel
        self.m_load = 0.

    # factory class method
    @classmethod
    def createPowerPlants(cls, pp_configs, fuels):
        return [cls(c['name'], PpType(c['type']), c['efficiency'], float(c['pmin']), float(c['pmax']), fuels[PpType(c['type'])]) \
                    for c in pp_configs ]

    def load(self):
        return round(10. * self.m_load) / 10.

    def reset_load(self):
        self.m_load = 0.

    def max_load(self):
        self.m_load = self.p_max()

    def is_underloaded(self):
        return self.m_load < self.m_p_min

    def add_load(self, load):
        dload = min(self.p_max() -  self.m_load, load)
        self.m_load += dload
        return dload

    def steal_load(self, load):
        dload = max(0., min(self.m_load - self.m_p_min, load))
        self.m_load -= dload
        return dload

    def p_max(self):
        return round(10. * self.m_fuel.phi() * self.m_p_max) / 10.

    def p_gen(self, load):
        return min(max(self.m_p_min, load), self.p_max()) # technically DelT * p_{min|max}

    def unit_cost(self):
        return self.m_fuel.total_cost(1., self.m_eta)

    def load_cost(self, load):
        return load * self.unit_cost()

    def cur_load_cost(self):
        return self.m_load * self.unit_cost()

    def budget_req(self, load):
        return self.m_fuel.total_cost(self.p_gen(load), self.m_eta)

    def budget_reserved(self):
        return self.budget_req(self.m_load)

class ProductionPlanner(object):
    def __init__(self, pl):
        self.m_log = logging.getLogger("ProductionPlanner")
# filters and lambdas: list(filter(lambda pp: pp['type'] == 'windturbine', pl['powerplants']))
        fuels = Fuel.createFuels(pl['fuels'])

        self.m_pps = PowerPlant.createPowerPlants(pl['powerplants'], fuels)

        self.m_exp_load = pl['load']
#        self.m_exp_load = 165
#        self.m_exp_load = 185
#        self.m_exp_load = 623
#        self.m_exp_load = 36 # Catch22! with wind(%)=10
        load_min = fn.reduce(op.add, [pp.p_max() for pp in list(filter(lambda pp: self.m_exp_load > pp.m_p_min, self.m_pps))], 0.)
        load_max = fn.reduce(op.add, [pp.p_max() for pp in self.m_pps])
        self.m_log.info(f"E[load]={self.m_exp_load} vs [min(load)={load_min}, max(load)={load_max}]")
        if self.m_exp_load > load_min:
            self.m_log.info("Catch22!")
        if load_max < self.m_exp_load:
            self.m_log.info("Once upon a time in Texas...")

        self.m_pp_mo_q = [(self.m_pps[i].unit_cost(), self.m_pps[i].p_max(), i, self.m_pps[i]) for i in range(0, len(self.m_pps))]
        heap.heapify(self.m_pp_mo_q)

    def display_pp(self, prefix, pp):
        self.m_log.debug(f"{prefix}{pp.m_name} {pp.m_type}")

    def display_merit_order(self): # to log or to dashboard
        for pp in heap.nsmallest(len(self.m_pps), self.m_pp_mo_q):
            self.m_log.info(f"{pp[3].m_name}: type={pp[3].m_type} unit cost={pp[0]} p_min={pp[3].m_p_min} p_max={pp[1]} load={pp[3].load()}")

    def try_recommit(self, load_to_recommit, plan_tp):
        initial_cost = plan_tp[0]
        prev_pp = plan_tp[1]

        res = self.__allocate_load(load_to_recommit, plan_tp, True, True)
        cost_recommitting = initial_cost - prev_pp.cur_load_cost() + res[0]

        self.m_log.info(f"Cost recommitting: {cost_recommitting}; uncovered load: {res[1]}")
        for pp_tp in self.m_pp_mo_q: pp_tp[3].reset_load();

        return (cost_recommitting, res[1])

    def try_stealing(self, load_to_steal, plan_tp):
        initial_cost = plan_tp[0]
        prev_pp = plan_tp[1]
        plan = plan_tp[2]

        res = self.__steal_load(load_to_steal, plan)
        cost_stealing = initial_cost + prev_pp.budget_reserved() - prev_pp.cur_load_cost() - res[0]

        self.m_log.info(f"Cost stealing: {cost_stealing}; uncovered load: {res[1]}")
        for i in range(2, len(plan) + 1): self.m_pps[plan[-i]].max_load()

        return (cost_stealing, res[1])

    def rebalance(self, plan_tp):
        if plan_tp[1] == None or not plan_tp[1].is_underloaded():
            return plan_tp

        cost = plan_tp[0]
        prev_pp = plan_tp[1]
        plan = plan_tp[2]
        self.m_log.info(f"Plan budget required: {cost}")
        self.display_pp("\tprev: ", prev_pp)
        committed_load = self.m_exp_load - prev_pp.load()
#        if committed_load < (prev_pp.m_p_min - prev_pp.load()): # we need to find a plant with a sufficiently lower minimum
#            self.m_log.info(f"#unloaded plants: {len(self.m_pp_mo_q)}; uncommitted: {prev_pp.load()}")
#            self.m_log.info("NYI!")
#            return (cost, prev_pp, plan)
#        else: # we can steal committed power to reach the minimum or recommit to a lower merit plant altogether, whatever is cheaper
        load_to_steal = round(10 * (prev_pp.m_p_min - prev_pp.load())) / 10
        load_to_recommit = prev_pp.load()
        self.m_log.info(f"p_min({prev_pp.m_p_min}): load to recommit {load_to_recommit} vs load to steal {load_to_steal}")

        recommit_result = self.try_recommit(load_to_recommit, plan_tp)
        steal_result = self.try_stealing(load_to_steal, plan_tp)

        if committed_load > 0 and (recommit_result[1] > 0. or steal_result[0] <= recommit_result[0]):
            self.m_log.info("Stealing...")
            res = self.__steal_load(load_to_steal, plan_tp[2])
            cost += prev_pp.budget_reserved() - prev_pp.cur_load_cost() - res[0]

            prev_pp.m_load = prev_pp.m_p_min
            tmp_pp = heap.heappop(self.m_pp_mo_q)[3]
            self.display_pp("\ttop: ", tmp_pp)
        elif steal_result[1] > 0. or steal_result[0] > recommit_result[0]:
            self.m_log.info("Recommitting")
            plan.pop()
            res = self.__allocate_load(load_to_recommit, plan_tp, True, False)
            cost += res[0] - prev_pp.cur_load_cost()

            prev_pp.reset_load()
            prev_pp = res[2]

        self.m_log.info(f"Adjusted budget: {cost}")

        return self.rebalance((cost, prev_pp, plan))
    
    def generate_plan(self):
        plan_tp = self.rebalance(self.__initial_commit())
        return (plan_tp[0], plan_tp[1], [self.m_pps[i] for i in plan_tp[2]])

    def __allocate_load(self, load, plan_tp, isRecommit, isTrial):
        prev_pp = plan_tp[1]
        top_pp = None
        cost = 0.
        try:
            pps = self.m_pp_mo_q.copy() if isRecommit else self.m_pp_mo_q
            while load > 0.:
                pp = pps[0][3]
                self.display_pp("\tscan: ", pp)
                # 1st order approximation; should be replaced with recursive search
                if (not isRecommit) or pp.unit_cost() > prev_pp.unit_cost() and pp.m_p_min <= prev_pp.m_p_min:
                    load -= pp.add_load(load)
                    cost += pp.cur_load_cost()
                    if not isTrial and pp.p_max() > 0.:
                        plan_tp[2].append(pps[0][2])
                top_pp = heap.heappop(pps)[3]
                self.display_pp("\ttop: ", top_pp)
        except IndexError:
            pass

        return (cost, load, top_pp)

    def __steal_load(self, load, plan):
        savings_steal =0.
        for i in range(2, len(plan) + 1):
            prev_cost = self.m_pps[plan[-i]].cur_load_cost()
            load -= self.m_pps[plan[-i]].steal_load(load)
            savings_steal += prev_cost - self.m_pps[plan[-i]].cur_load_cost()
            if load <= 0.:
                break

        return (savings_steal, load)

    def __initial_commit(self):
        plan_tp = (0., None, [])
        res = self.__allocate_load(self.m_exp_load, plan_tp, False, False)
        if res[1] > 0.:
            self.m_log.info("Underpowered!\nNo exception handling yet. Let it crash...\nNYI!")
            sys.exit(1)
        return (res[0], res[2], plan_tp[2])

def test_plan(payload_uri):
    log = logging.getLogger(__name__)
    log.info(f"--- {payload_uri} ---")
    pl_file = open(payload_uri, "r")
    planner = ProductionPlanner(json.load(pl_file))
    plan_tp = planner.generate_plan()

    commit_state =  f" with a load of {plan_tp[1].load()} still uncommitted:" if plan_tp[1].is_underloaded() \
                    else "; load fully committed"
    log.info(f"\nPlan at total cost of {plan_tp[0]}{commit_state}")
    for pp in plan_tp[2]:
        log.info(f"{pp.m_name}: type={pp.m_type} unit cost={pp.unit_cost()} load={pp.load()}")
    log.info("---------------------")

def main(): # temp for test until we have a controller for a web framework
    test_plan("payload1.json")
    test_plan("payload2.json")
    test_plan("payload3.json")

if __name__ == "__main__":
    main()

from QuantLib import  VanillaOption, Option, EuropeanExercise, Days, Period, PlainVanillaPayoff # type: ignore

from app.data_gen.quantlib_setup.models import OptionParams


def get_option_with_bates(option_params: OptionParams) -> VanillaOption:
    
    # Option definition
    payoff = PlainVanillaPayoff(Option.Call, option_params.strike)
    exercise = EuropeanExercise(date=option_params.evaluation_date + Period(int(option_params.tau), Days))
    option = VanillaOption(payoff, exercise)

    option.setPricingEngine(option_params.engine)
    return option


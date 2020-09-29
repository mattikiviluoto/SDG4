#!/usr/bin/env python3

import numpy as np
import pandas as pd
from collections import defaultdict

class Reader():
    """ A class for parsing data from CIA-factbook .jsons with few accessor methods. """

    def t_e_wrapper(self, country, function, *args, **kwargs):
        """ This wrapper function is necessary to handle exceptions caused by missing data. """ 
        try:
            look = function(*args, **kwargs)                    # Evaluate query
            return look 
        except KeyError as missing:
            self.missing_data[country].append(missing.args[0])  # Add the missing data into appropriate slot.
            return np.nan                                       # Add a NaN as a value for the missing key.

    def encode(self):
        """ This creates a dictionary which encodes query words to match the nested structure of the factbook.json format. """
        return {
            # Geographic data
            "continent": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["geography"]["map_references"]) for x in self.countries],
            "area": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["geography"]["area"]["total"]["value"]) for x in self.countries],
            "irrigated": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["geography"]["irrigated_land"]["value"]) for x in self.countries],
            # Population data
            "population": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["population"]["total"]) for x in self.countries],
            "children": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["age_structure"]["0_to_14"]["percent"]) for x in self.countries],
            "median_age": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["median_age"]["total"]["value"]) for x in self.countries],
            "population_growth": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["population_growth_rate"]["growth_rate"]) for x in self.countries],
            "birth_rate": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["birth_rate"]["births_per_1000_population"]) for x in self.countries],
            "death_rate": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["death_rate"]["deaths_per_1000_population"]) for x in self.countries],
            "migration": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["net_migration_rate"]["migrants_per_1000_population"]) for x in self.countries],
            "infant_mortality": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["infant_mortality_rate"]["total"]["value"]) for x in self.countries],
            "life_expectancy": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["life_expectancy_at_birth"]["total_population"]["value"]) for x in self.countries],
            "fertility": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["total_fertility_rate"]["children_born_per_woman"]) for x in self.countries],
            "literacy": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["literacy"]["total_population"]["value"]) for x in self.countries],
            "lit_men": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["literacy"]["male"]["value"]) for x in self.countries],
            "lit_women": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["people"]["literacy"]["female"]["value"]) for x in self.countries],
            # Economic data
            "growth": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["gdp"]["real_growth_rate"]["annual_values"][0]["value"]) for x in self.countries],
            "gdp": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["gdp"]["per_capita_purchasing_power_parity"]["annual_values"][0]["value"]) for x in self.countries],
            "agriculture": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["gdp"]["composition"]["by_sector_of_origin"]["sectors"]["agriculture"]["value"]) for x in self.countries],
            "industry": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["gdp"]["composition"]["by_sector_of_origin"]["sectors"]["industry"]["value"]) for x in self.countries],
            "services": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["gdp"]["composition"]["by_sector_of_origin"]["sectors"]["services"]["value"]) for x in self.countries],
            "unemployment": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["unemployment_rate"]["annual_values"][0]["value"]) for x in self.countries],
            "poverty": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["population_below_poverty_line"]["value"]) for x in self.countries],
            "low_decile": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["household_income_by_percentage_share"]["lowest_ten_percent"]["value"]) for x in self.countries],
            "high_decile": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["household_income_by_percentage_share"]["highest_ten_percent"]["value"]) for x in self.countries],
            "revenues": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["budget"]["revenues"]["value"]) for x in self.countries],
            "expenditures": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["budget"]["expenditures"]["value"]) for x in self.countries],
            "public_debt": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["public_debt"]["annual_values"][0]["value"]) for x in self.countries],
            "inflation": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["inflation_rate"]["annual_values"][0]["value"]) for x in self.countries],            
            "reserves": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["reserves_of_foreign_exchange_and_gold"]["annual_values"][0]["value"]) for x in self.countries],
            "foreign_debt": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["economy"]["external_debt"]["annual_values"][0]["value"]) for x in self.countries],
            # Military data
            "military": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["military_and_security"]["expenditures"]["annual_values"][0]["value"]) for x in self.countries],
            # Transnational issues
            "refugees": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["transnational_issues"]["refugees_and_iternally_displaced_persons"]["refugees"]["by_country"][0]["people"]) for x in self.countries],            
            "internal_refugees": [self.t_e_wrapper(x, lambda : self.factbook[x]["data"]["transnational_issues"]["refugees_and_iternally_displaced_persons"]["internally_displaced_persons"]["people"]) for x in self.countries]
            }

    def __init__(self, factbook):
        """ Constructs the Reader object from a given .json file """
        self.factbook = factbook
        self.countries = list(factbook.keys())   # a list of all entries in the factbook 
        self.missing_data = defaultdict(list)    # a dictionary with countries as keys and list of missing information as values
        self.data = self.encode()

    def get_missing_data(self, country):    
        """ See what data is missing for a given country. """
        return self.missing_data[country]

    def read_data(self, which):
        """ Takes a list of all information to be fetched as an argument. 
        The elements of the list must match keys found in the dictionary self.data.
        Returns a data frame with countries as indexes and data as columns."""

        data_array = np.array([self.data[x] for x in which])
        df = pd.DataFrame(data_array.T, columns=which, index=self.countries) # read info we are interested in 
        return df


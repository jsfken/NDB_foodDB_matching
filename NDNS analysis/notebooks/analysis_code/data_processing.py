import pandas as pd
import numpy as np
import regex as re
from rpy2.robjects.vectors import StrVector, FloatVector, IntVector
from rpy2.robjects import DataFrame
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
import time


  
def print_columns(df: pd.DataFrame):
    
    col_list = []
    
    for col in df.columns:
        col_list.append(col)
        print(f"'{col}',")
    
    return col_list

def search_substrings(input_list: list, search_terms: list):
    
    """
    Goes through each string in input_list and returns a list of items that contain any of the substrings provided in the list search_terms
    """
    
    substring_matches = [i for i in input_list if any(j.lower() in i.lower() for j in search_terms)]
    
    return substring_matches


def add_env_data_ndns(row: pd.Series, 
                      NDB_data: pd.DataFrame, 
                      ndns_ndb_mapping: pd.DataFrame, 
                      df_impacts: pd.DataFrame, 
                      mean_env_columns: list, 
                      error_columns: list,
                      conversion_dict: dict,
                      sensitivity_analysis = False):

  food_code = row['FoodNumber']
  food_desc = row['FoodName']

  # list of ndb items for a given food code

  # Check if the food code exists in the list of original NDB items
  ndb_items = list(NDB_data[NDB_data['Food composition record ID']==food_code]['Local description'])
  if len(ndb_items)==0:
    # If not, use the mapping to find the equivilent NDB item for that food code
    ndb_items = ndns_ndb_mapping[ndns_ndb_mapping['FoodNumber']==food_code]['matched'].tolist()

  # compute the combined impacts of each NDB item
  impacts = df_impacts[df_impacts.index.isin(ndb_items)]
  mean_impacts = impacts[mean_env_columns].mean(skipna=True)
  error_impacts = ((impacts[error_columns]**2).sum(skipna=True))**0.5/len(ndb_items)

  try:
    ndb_desc = NDB_data[NDB_data['Food composition record ID']==food_code]['Local description'].iloc[0]
  except IndexError:
    try:
      ndb_desc = ndns_ndb_mapping[ndns_ndb_mapping['NDNSFoodName']==food_desc]['matched'].iloc[0]
    except IndexError:
      return row
      
  # if the food item is an item that does not account for diluted water seperately
  if food_desc in conversion_dict.keys():
    if sensitivity_analysis:
        CF = conversion_dict[food_desc]
        CF += np.random.normal(loc=0, scale = 0.2*CF)
    
        row[mean_env_columns] = (mean_impacts/100)*row['TotalGrams']*CF
        row[error_columns] = (error_impacts/100)*row['TotalGrams']*CF
        
    else:
        
        row[mean_env_columns] = (mean_impacts/100)*row['TotalGrams']*conversion_dict[food_desc]
        row[error_columns] = (error_impacts/100)*row['TotalGrams']*conversion_dict[food_desc]
    
        # Include the water use from the remaining gram weight of the converted item and convert to litres
        row['median_WaterUse'] += (1-conversion_dict[food_desc])*row['TotalGrams']*0.001 
        row['mean_WaterUse'] += (1- conversion_dict[food_desc])*row['TotalGrams']*0.001
  else:
    # add the impacts of each item weighted by consumption
    row[mean_env_columns] = (mean_impacts/100)*row['TotalGrams'] ## impact is per 100g
    row[error_columns] = (error_impacts/100)*row['TotalGrams']

  return row


def add_sw(row:pd.Series, ind_data:pd.DataFrame):

  id = row['Cpseriala']
  SW = ind_data[ind_data['Cpseriala']==id]['SHeS_Intake24_wt_sc'].iloc[0]

  return SW
  
  



################# Baseline indicators #######################

# define a function to convert a string to a float if it contains a float
def convert_to_float(s):
    pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
    if pattern.match(s):
        return float(s)
    else:
        return s

#from pandas.core.groupby.ops import Int64Dtype
def add_variable(row, variable, dem_data):

    id = float(row.name)
    ind = dem_data[dem_data['seriali']==id]
    value = ind[variable].values[0]

    if variable == 'SIMD20_RPa' or variable == 'Ethnic05':
      value = str(value)
    else:

      if value == 'Male':
        value=0
      elif value == 'Female':
        value=1
      elif isinstance(value, str):
        value = convert_to_float(value)
      else:
        pass

      if isinstance(value, str):
        value = np.nan

    return value


def HIncome(df):
    # Creating new columns based on SIMD_labels
  income_mapping = {
       1: ('HIncome lowest tertile', 1),
       2: ('HIncome middle tertile', 1),
       3: ('HIncome highest tertile', 1),
      
  }

  for new_col in ['HIncome lowest tertile', 'HIncome middle tertile', 'HIncome highest tertile']:
      df[new_col] = 0

  for index, row in df.iterrows():
      label = row['HIncome_tertiles']
      if label in income_mapping:
          col, value = income_mapping[label]
          df.at[index, col] = value

  return df

def Ethnicity(df):

    # Creating new columns based on SIMD_labels
  eth_mapping = {
      1: ('White', 1),
      2: ('Mixed ethnic group', 1),
      3: ('Black or Black British', 1),
      4: ('Asian or asian British', 1),
      5: ('Any other group', 1)
  }

  for new_col in ['White', 'Mixed ethnic group', 'Black or Black British', 'Asian or asian British', 'Any other group']:
      df[new_col] = 0

  for index, row in df.iterrows():
      label = row['Ethnicity']
      if label in eth_mapping:
          col, value = eth_mapping[label]
          df.at[index, col] = value

  return df
  
def region(df):

    # Creating new columns based on SIMD_labels
  region_mapping = {
      1: ('England: North', 1),
      2: ('England: Central/Midlands', 1),
      3: ('England: South (incl. London)', 1),
      4: ('Scotland', 1),
      5: ('Wales', 1),
      6: ('Northern Ireland', 1)
  }

  for region_name, _ in region_mapping.values():
    df[region_name] = 0

  for index, row in df.iterrows():
      label = row['region']
      if label in region_mapping:
          col, value = region_mapping[label]
          df.at[index, col] = value

  return df

def add_non_diet_variables(df: pd.DataFrame, dem_data: pd.DataFrame):

  df['Sample Weight'] = df.apply(lambda row: add_variable(row, variable='wti_Y911', dem_data=dem_data), axis=1)
  df['astrata1'] = df.apply(lambda row: add_variable(row, variable='astrata1', dem_data=dem_data), axis=1)
  df['astrata2'] = df.apply(lambda row: add_variable(row, variable='astrata2', dem_data=dem_data), axis=1)
  df['astrata3'] = df.apply(lambda row: add_variable(row, variable='astrata3', dem_data=dem_data), axis=1)
  df['astrata4'] = df.apply(lambda row: add_variable(row, variable='astrata4', dem_data=dem_data), axis=1)
  df['astrata5'] = df.apply(lambda row: add_variable(row, variable='astrata5', dem_data=dem_data), axis=1)
  df['psu'] = df.apply(lambda row: add_variable(row, variable='Area', dem_data=dem_data), axis=1)
  df['age'] = df.apply(lambda row: add_variable(row, variable='AgeR', dem_data=dem_data), axis=1)
  df['Sex'] = df.apply(lambda row: add_variable(row, variable='Sex', dem_data=dem_data), axis=1)

  df['HIncome_tertiles'] = df.apply(lambda row: add_variable(row, variable='eqv3', dem_data=dem_data), axis=1)
  df['Ethnicity'] = df.apply(lambda row: add_variable(row, variable='ethgrp5', dem_data=dem_data), axis=1)
  df['region'] = df.apply(lambda row: add_variable(row, variable='region', dem_data=dem_data), axis=1)

  df = HIncome(df)
  df = Ethnicity(df)
  df= region(df)

  return df
  
def nutrient_intake(row: pd.Series, diet_data: pd.DataFrame, nutrients: list, error_columns: list):

    id = float(row.name)
    diet = diet_data[diet_data['seriali'] == id]
    num_days = float(diet['DiaryDaysCompleted'].unique())

    for nutr in nutrients:
        if nutr in error_columns:
            row[nutr] = np.sqrt((diet[nutr] ** 2).sum()) / num_days
        else:
            row[nutr] = float(diet[nutr].sum()) / num_days

    return row


def filter_dataframes(dfs: list, conditions_tuple: tuple):
    """
    Apply multiple filters to a list of DataFrames based on given conditions.

    Parameters:
    dfs (list): List of DataFrames to apply the filters on.
    conditions (list): List of dictionaries containing conditions.

    Returns:
    list: List of filtered DataFrames.
    """

    desc = conditions_tuple[0]
    conditions = conditions_tuple[1]

    if desc == 'Overall':
      return dfs
    else:
      filtered_dfs = []
      for df in dfs:
          filtered_df = pd.DataFrame()
          for idx, cond in enumerate(conditions):
              column = cond['column']
              condition = cond['condition']
              boolean_operator = cond['boolean_operator']
              if idx == 0:
                  filtered_df = df[df[column].apply(condition)]
              elif boolean_operator == 'and':
                  filtered_df = filtered_df[filtered_df[column].apply(condition)]
              elif boolean_operator == 'or':
                  filtered_df = pd.concat([filtered_df, df[df[column].apply(condition)]], ignore_index=True).drop_duplicates()

          filtered_dfs.append(filtered_df)
      return filtered_dfs
      
      
def pandas_to_r_dataframe(df: pd.DataFrame):
    """
    Converts a pandas DataFrame to an R-compatible DataFrame manually.

    Args:
        df (pd.DataFrame): The pandas DataFrame to convert.

    Returns:
        rpy2.robjects.DataFrame: An R-compatible DataFrame.
    """


    # Create a dictionary of R-compatible vectors
    r_data = {}
    for column_name, column_data in df.items():
        if pd.api.types.is_numeric_dtype(column_data):
            r_data[column_name] = FloatVector(column_data)
        elif pd.api.types.is_integer_dtype(column_data):
            r_data[column_name] = IntVector(column_data)
        elif pd.api.types.is_string_dtype(column_data):
            r_data[column_name] = StrVector(column_data)
        else:
            raise ValueError(f"Unsupported column type for '{column_name}'.")

    # Return the R DataFrame
    return DataFrame(r_data)

# Calculate weighted mean within strata
def weighted_mean(group: pd.DataFrame, variable: str):
    return np.average(group[variable], weights=group['Sample Weight'])
    
    
def survey_se(df: pd.DataFrame, variable: str):
    """
    Runs R code to calculate standard error for a given column in a survey design.
    Tries multiple stratification variables until survey design succeeds.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        variable (str): The column for which to compute confidence intervals.

    Returns:
        tuple: (mean, standard_error)
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
        
    print(variable)

    # Rename for R compatibility
    df_renamed = df.rename(columns={'Sample Weight': 'sw'})
    r_df = pandas_to_r_dataframe(df_renamed)
    r.assign("data", r_df)

    r('colnames(data) <- make.names(colnames(data))')

    # Set the variable name in R
    r.assign("column_name", variable)

    # R code to try multiple strata variables, dynamically building psu without rlang
    r("""
    library(survey)
    library(dplyr)
    
    strata_candidates <- c("astrata1", "astrata2", "astrata3", "astrata4", "astrata5")
    design <- NULL
    
    for (strata_var in strata_candidates) {
      if (!strata_var %in% names(data)) next
    
      
    
      try({
        data[[strata_var]] <- as.character(data[[strata_var]])
        data$psu <- as.character(data$psu)
        data$psu <- interaction(data[[strata_var]], data$psu, sep = "_")
    
        design_attempt <- svydesign(
          id = ~psu,
          strata = as.formula(paste0("~", strata_var)),
          weights = ~sw,
          data = data,
          nest = TRUE
        )
    
        # Test if estimation will succeed
        ci_test <- try(svymean(as.formula(paste("~", column_name)), design_attempt), silent = TRUE)
    
        if (!inherits(ci_test, "try-error")) {
          design <- design_attempt
          message("Successfully created survey design with: ", strata_var)
          break
        } 
    
      }, silent = TRUE)
    }
    
    if (is.null(design)) stop("Failed to create a valid survey design with any strata variable.")
    
    ci_result <- svymean(as.formula(paste("~", column_name)), design)
    mean_value <- coef(ci_result)[1]
    se_value <- SE(ci_result)[1]

    """)

    mean_value = r("mean_value")[0]
    se_value = r("se_value")[0]

    return mean_value, se_value


# def survey_se(df: pd.DataFrame, variable: str):
#     """
#     Runs R code to calculate standard error for a given column in a survey design.

#     Args:
#         df (pd.DataFrame): The input pandas DataFrame.
#         variable (str): The column for which to compute confidence intervals.

#     Returns:
#         tuple: (lower_confidence_interval, upper_confidence_interval)
#     """
#     # Ensure the DataFrame is compatible with rpy2
#     if not isinstance(df, pd.DataFrame):
#         raise ValueError("Input data must be a pandas DataFrame.")

#     df_renamed = df.rename(columns={'Sample Weight': 'sw'})

#     # Manually convert the pandas DataFrame to an R-compatible DataFrame
#     r_df = pandas_to_r_dataframe(df_renamed)

#     # Assign the R DataFrame to a variable in the R environment
#     r.assign("data", r_df)

#     # Clean column names in R to make them valid R identifiers
#     r('colnames(data) <- make.names(colnames(data))')

#     # Create unique PSU identifiers (adjust for your dataset structure)
#     r("""
#     library(dplyr)
#     data <- data %>% mutate(psu = interaction(strata, psu, sep = "_"))
#     """)

#     # Define the survey design in R
#     r("""
#     design <- svydesign(
#         id = ~psu,
#         strata = ~strata,
#         weights = ~sw,
#         data = data,
#         nest = TRUE
#     )
#     """)

#     # Compute the confidence intervals in R
#     r.assign("column_name", variable)

#     r("""
#     ci_result <- svymean(as.formula(paste("~", column_name)), design)
#     mean_value <- coef(ci_result)[1]  # Extract the mean
#     se_value <- SE(ci_result)[1]      # Extract the standard error
#     """)

#     # Extract mean and standard error
#     mean_value = r("mean_value")[0]  # Convert mean to Python float
#     se_value = r("se_value")[0]

#     return mean_value, se_value
    
    
def results_table_demographic_group(
                                    dem_group_dict: dict, 
                                    df_baseline: pd.DataFrame, 
                                    indicators: list,
                                    mean_env_columns: list,
                                    error_columns:list, 
                                    ):

  """
  Calculates the per capita values for all indicators in the list indicators in the demographic group specified by dem_group_dict

  """

  start_time = time.time()
  columns = [f"{dem_group}" for dem_group in dem_group_dict['dem_groups'].keys()]

  df_results = pd.DataFrame(index=indicators, columns = columns)

  for dem_group in dem_group_dict['dem_groups'].keys():
      print(dem_group)
      condition = dem_group_dict['dem_groups'][dem_group]
      #condition = condition_dict[dem_group]
      filtered_data = filter_dataframes(dfs=[df_baseline], conditions_tuple = condition)  # conditions are part of a tuple, with the first element being the description and the second being the set of conditions
    
      df_baseline_demgroup = filtered_data[0]
      print(dem_group, len(df_baseline_demgroup))
      
      for indicator in indicators:
        mean_baseline, survey_error_baseline = survey_se(df=df_baseline_demgroup, variable=indicator)
      
        if indicator in mean_env_columns:
            if 'price' in indicator:
              error_column = 'sd_price_sim'
            else:
              error_column = f"sd_{indicator}"
    
            within_item_error_baseline =  (1/df_baseline_demgroup['Sample Weight'].sum()) * np.sqrt( ((df_baseline_demgroup[error_column].multiply(df_baseline_demgroup['Sample Weight'], axis=0))**2).sum())
       
        else:
            within_item_error_baseline = 0
            within_item_error_scenario = 0
    
        error_baseline = np.sqrt(within_item_error_baseline**2 + survey_error_baseline**2)
        mean_baseline = np.round(mean_baseline, 2)
        
      
        lower_baseline = np.round(mean_baseline - 1.96*error_baseline, 2)
        upper_baseline = np.round(mean_baseline + 1.96*error_baseline, 2)
        df_results.loc[indicator, dem_group] = f"{mean_baseline}, ({lower_baseline}, {upper_baseline})"
        print(dem_group, f"{mean_baseline}, ({lower_baseline}, {upper_baseline})")
        

  end_time = time.time()
  execution_time = end_time - start_time

  return df_results


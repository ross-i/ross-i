
# Given user-specified parameters, this script pulls sales data and analyzes whether the data is sufficient
#       to forecast future sales in the given segments.

# July-August 2022
# Author: Matthew Rossi

from LibRefDAO.DAO import DAO

import os
import os.path
import uuid
import importlib
import time
import datetime

import pyspark
from pyspark.sql import Row
import pyspark.sql.functions as func
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

def autofill_config(industry_dict):
    """ Supplies default values for industry_dict if missing """
    
    # Classifier should be a list of strings. If it's a string, make it a list with a single element.
    if isinstance(industry_dict["classifier"], str):
        industry_dict["classifier"] = [industry_dict["classifier"]]
    # Do the same for "groups".
    if "groups" in industry_dict and isinstance(industry_dict["groups"], str):
        industry_dict["groups"] = [industry_dict["groups"]]
    
    if "outlet_lu" not in industry_dict:
        industry_dict["outlet_lu"] = "_".join(industry_dict["fact_table"].split("_")[:2]) + "_outlet_lu"
    
    # By default, don't break down to attribute level.
    if "attribute_level" not in industry_dict:
        industry_dict["attribute_level"] = False
    industry_dict["attr_cols"] = []
    if industry_dict["attribute_level"]:
        industry_dict["attr_cols"] = ["attr_desc", "attrvl_desc"]
        if "attrval_lu" not in industry_dict:
            industry_dict["attrval_lu"] = "_".join(industry_dict["fact_table"].split("_")[:2]) + "_attributes_value"   
    
    # Create a dictionary that maps classifiers to the name of classifier lookup tables.
    if "classifier_lu" not in industry_dict:
        lib_code = "_".join(industry_dict["fact_table"].split("_")[:2])
        industry_dict["classifier_lu"] = {clf:lib_code + "_" + clf + "_lu" for clf in industry_dict["classifier"]}  
    
    # A list of column names that alternates between <classifier> and <classifier>_name.
    # This list is used to order columns so that coded values appear next to the name value pulled from a lookup table.
    clf_cols_list = [[clf, clf+"_name"] for clf in industry_dict["classifier"]]
    clf_cols_list = [col for col_pair in clf_cols_list for col in col_pair]
    industry_dict["clf_cols"] = clf_cols_list
    
    # Handle monthly/weekly data, setting feasibility requirements and lag accordingly.
    if "time_var" not in industry_dict:
        industry_dict["time_var"] = 'ppmonth' if "Month" in industry_dict["library"] else 'ppweek'
    if "feasibility_criteria" not in industry_dict:
        hist_req = 104 if industry_dict["time_var"] =='ppweek' else 24
        volume_req = 5000 if industry_dict["time_var"] =='ppweek' else 20000
        industry_dict["feasibility_criteria"] = (hist_req, volume_req, 0.2)
    if "lag" not in industry_dict:
        buffer = 7 if industry_dict["time_var"] =='ppweek' else 4
        industry_dict["lag"] = industry_dict["feasibility_criteria"][0] + buffer
    
    # More default settings.
    if "unit_var" not in industry_dict:
        industry_dict["unit_var"] = "proj_unitssold"
    if "dollar_var" not in industry_dict:
        industry_dict["dollar_var"] = "proj_totalvalue"
        
    # Default releasability criteria.
    if "releasability_criteria" not in industry_dict:
        industry_dict["releasability_criteria"] = (3, 75)
    # If you set releasability criteria to "None", then set values such that all segments will pass.
    elif industry_dict["releasability_criteria"] is None:
        industry_dict["releasability_criteria"] = (1, 101)
        
    # Use a UUID + date if no job_id was given
    if industry_dict.get("job_id", None) is None:
        date = str(datetime.date.today()).replace("-", "")
        uuid_str = str(uuid.uuid4())[0:4] 
        industry_dict["job_id"] = uuid_str + "_" + date
        print("Using job_id '{}'".format(industry_dict["job_id"]))
    
    # None values of the filters will break the query -- change to empty string
    if industry_dict.get("extra_filter", None) is None:
        industry_dict["extra_filter"] = ""
    if industry_dict.get("attribute_filter", None) is None:
        industry_dict["attribute_filter"] = ""
        
    # If filters were inputted, they should begin with "and"
    if industry_dict["extra_filter"] and industry_dict["extra_filter"].strip().lower()[0:3] != "and":
        industry_dict["extra_filter"] = "and " + industry_dict["extra_filter"]
    if industry_dict["attribute_filter"] and industry_dict["attribute_filter"].strip().lower()[0:3] != "and":
        industry_dict["attribute_filter"] = "and " + industry_dict["attribute_filter"]
    
    return industry_dict
  
def check_config_values(industry_dict):
    """ Raises a ValueError if an important key wasn't in the industry dictionary 
        Also warns user if certain keys appear to have the wrong values
    """
    # Check for missing keys
    essential_keys = ["library", "job_id", "fact_table", "outlet_lu",
                     "classifier", "classifier_lu", "time_var",
                     "lag", "unit_var", "dollar_var"]
    actual_keys = list(industry_dict.keys())
    for key in essential_keys:
        if key not in actual_keys:
            raise ValueError("Required key '{}' was not provided in the industry dictionary".format(key))
    
    # Check for weird key values
    if industry_dict["time_var"] == "ppmonth" and len(str(industry_dict["lag"])) > 2:
        print("Warning: time_var is set to ppmonth and lag is set to {}. This may be too long for monthly data".format(industry_dict["lag"]))
    if industry_dict["time_var"] == "ppweek" and len(str(industry_dict["lag"])) < 3:
        print("Warning: time_var is set to ppweek and lag is set to {}. This may be too short for weekly data".format(industry_dict["lag"]))
    if "unitssold" not in industry_dict["unit_var"]:
        print("Warning: unexpected unit_var '{}'".format(industry_dict["unit_var"]))
    if "totalvalue" not in industry_dict["dollar_var"]:
        print("Warning: unexpected dollar_var '{}'".format(industry_dict["dollar_var"]))
    for clf in industry_dict["classifier"]:
        if clf not in industry_dict["classifier_lu"][clf]:
            print(f"""Warning: classifier_lu {industry_dict["classifier_lu"][clf]} doesn't contain the word '{clf}'""")
    
    # Check that all our table names share the same code that corresponds to the library.
    tables = [industry_dict["fact_table"],industry_dict["outlet_lu"], *industry_dict["classifier_lu"].values()]
    if industry_dict["attribute_level"]:
        tables.append(industry_dict["attrval_lu"])
    shared_text = [table.split("_")[1] for table in tables]
    if len(set(shared_text)) != 1:
        print("Warning: table names may be incorrect {}".format(shared_text))

def extract_sales_data(dao, industry_dict):
    """ Fetches raw sales data. Output is a spark dataframe with columns [time_var, outlet, [classifiers], itemid, units, dollars] """
    
    print("Fetching sales data...", end="")
    
    # Get the largest value of ppweek or ppmonth so we can pull the most recent data.
    max_time_query = f"""select max({industry_dict["time_var"]}) as time_var from rawdata.{industry_dict["fact_table"]}"""
    max_time = dao.getData(max_time_query).collect()[0][0]

    # Pull raw sales data, with all our classifier columns included.
    classifiers_str = ",".join(industry_dict["classifier"])
    sales_query = f"""
    select distinct {industry_dict["time_var"]},
                    outlet,
                    {classifiers_str},
                    itemid,
                    (sum({industry_dict["unit_var"]})) as units,
                    (sum({industry_dict["dollar_var"]})) as dollars
    from rawdata.{industry_dict["fact_table"]}
    where ({industry_dict["time_var"]} between {max_time}-{industry_dict["lag"]} and {max_time}) {industry_dict["extra_filter"]}
    group by
            {industry_dict["time_var"]},
            outlet,
            {classifiers_str},
            itemid
    """
    sales_df = dao.getData(sales_query)
    
    # Fetch outlet names and join to sales dataframe.
    outlet_vals_str = ",".join(map(str,list(sales_df.select('outlet').distinct().toPandas()['outlet']))) #convert the list of outlets into a single, comma-separated string for use in our query
    outlet_lu_query = f"""
    select outlet, outlet_name
    from rawdata.{industry_dict["outlet_lu"]}
    where outlet in ({outlet_vals_str})
    """
    sales_df = sales_df.join(dao.getData(outlet_lu_query), 'outlet', 'inner').select(industry_dict["time_var"], 'outlet_name', *industry_dict["classifier"], 'itemid', 'units', 'dollars')
    
    print("Done.")
    
    return sales_df

def extract_item_data(dao, sales_df, industry_dict):
    """ Fetches item dims data. Output is a spark dataframe with columns [itemid, [classifier, classifier_name]* [attr_desc], [attrvl_desc]] """
    
    print("Fetching item dimensions...", end="")
    
    # First, fetch all classifier lookup tables, filtering only the classifier values that appear in our sales data. Append them to a dictionary.
    clf_lus = {}
    for clf in industry_dict["classifier"]:
        clf_vals_str = ",".join(map(str,list(sales_df.select(clf).distinct().toPandas()[clf]))) #convert the list of classifier values into a single, comma-separated string for use in our query
        clf_lu_query = f"""
        select {clf}, {clf+"_name"}
        from rawdata.{industry_dict["classifier_lu"][clf]}
        where {clf} in ({clf_vals_str})
        """
        clf_lus[clf] = dao.getData(clf_lu_query)

    # Next, create a single-column df containing all unique itemids.
    item_dims_df = sales_df.select('itemid', *industry_dict["classifier"]).distinct().sort('itemid')
    
    # Next, if requested, fetch attribute data and join to the dataframe.
    attr_cols = []
    if industry_dict["attribute_level"]:
        attr_cols = ["attr_desc", "attrvl_desc"]
        itemids_list_str = ",".join(map(str,list(item_dims_df.select('itemid').toPandas()['itemid']))) #convert the list of itemids into a single, comma-separated string for use in our query
        item_dims_query = f"""
        select distinct itemid, attr_desc, attrvl_desc
        from rawdata.{industry_dict["attrval_lu"]}
        where itemid in ({itemids_list_str}) {industry_dict["attribute_filter"]}
        """
        item_dims_df = dao.getData(item_dims_query).sort('itemid').join(item_dims_df, 'itemid', 'right')
    
    # Perform joins.
    for clf in industry_dict["classifier"]:
        item_dims_df = item_dims_df.join(clf_lus[clf], clf, 'inner')
    
    print("Done.")
    
    return item_dims_df.select('itemid', *industry_dict["clf_cols"], *attr_cols)
    
def prep_data(industry_dict, sales_df, items_df):
    """ Joins and aggregates dataframes. 
        Also appends observations at the total classifier (ex. category or subcategory) level for all hierarchy levels except the lowest.
    """
    
    # Prepare the week-classifier-outlet-attribute dataset 
    print("Transforming...", end="")
    merged_df = sales_df.join(items_df, how='inner', on=['itemid', *industry_dict["classifier"]])
    merged_df = merged_df.groupby(industry_dict["time_var"], *industry_dict["clf_cols"], "outlet_name", *industry_dict["attr_cols"])\
                  .agg(func.sum('dollars').alias('dollars'), func.sum('units').alias('units'))
    
    # Prepare the total classifier level dataset, used for calculating feasibility at the total classifier level (i.e no attributes).
    # Calculate totals for all classifiers, but omit the last if we aren't using attribute level data.
    clfs_to_total = industry_dict["classifier"] if industry_dict["attribute_level"] else industry_dict["classifier"][:-1]
    for clf in clfs_to_total:
        classifier_lu = items_df.select(clf, clf+"_name").distinct()
        merged_df_total = sales_df.join(classifier_lu, clf, 'inner')
        merged_df_total = merged_df_total.groupby(industry_dict["time_var"], clf, clf+"_name", "outlet_name")\
            .agg(func.sum('dollars').alias('dollars'), func.sum('units').alias('units'))
        
        # Add fake columns for every other classifier.
        for classifier in industry_dict["classifier"]:
            if classifier == clf:
                continue
            merged_df_total = merged_df_total.withColumn(classifier, func.concat(func.lit("Total "), func.col(clf+"_name")))
            merged_df_total = merged_df_total.withColumn(classifier+"_name", func.concat(func.lit("Total "), func.col(clf+"_name")))
        
        # Add fake attr_desc/attrvl_desc columns 
        if industry_dict["attribute_level"]:  
            merged_df_total = merged_df_total.withColumn("attr_desc", func.concat(func.lit("Total "), func.col(clf+"_name")))
            merged_df_total = merged_df_total.withColumn("attrvl_desc", func.concat(func.lit("Total "), func.col(clf+"_name")))

        merged_df_total = merged_df_total.select(merged_df.columns)
        
        # Now stack both datasets
        merged_df = merged_df_total.union(merged_df)
    
    # Drop encoded classifier columns since we have the names.
    merged_df = merged_df.drop(*industry_dict["classifier"])
    
    print("Done.")
    
    return merged_df

def is_releasable(df, outlet_var="outlet_name", value_var="units", num_outlets_required=3, largest_share_allowed=75):
    """ Returns True if data are releasable based on values specified in num_outlets_required/largest_share_allowed """
    
    df_filtered = df[df[value_var] >= 0]
    df_aggregated = df_filtered[[outlet_var, value_var]].groupby(outlet_var).sum().reset_index().sort_values(by=value_var, ascending=False)
    
    # Add share
    total_units = df_aggregated[value_var].sum()
    df_aggregated[outlet_var+"_share"] = round(100*(df_aggregated[value_var]/total_units), 2)
    
    if df_aggregated.drop_duplicates().count()[outlet_var] < num_outlets_required:
        releasable = False
        reason_string = "Too few outlets selling ({} required)".format(num_outlets_required)
    elif df_aggregated.iloc[0, 2] > largest_share_allowed:
        releasable = False
        outlet = df_aggregated.iloc[0, 0]
        share = df_aggregated.iloc[0, 2]
        reason_string = "Outlet '{}' has {}% share, exceeding the allowed threshold of {}%".format(outlet, share, largest_share_allowed)
    else:
        releasable = True
        reason_string = "Releasability criteria met (number of oulets >= {} and maximum share <= {}%)".format(num_outlets_required, largest_share_allowed)
    
    return releasable, reason_string

def is_feasible(df, industry_dict, num_time_units_required=None, avg_units_required=None, seas_score_required=0.20):
    """ Returns True if the data are considered feasible based on a decision rule using average units per week, 
        number of weeks sold, and seasonality score (a measure of how seasonal the data are)"""
    
    # If unspecified, require 2 years of data
    time_unit = "week" if industry_dict["time_var"] == "ppweek" else "month"
    if num_time_units_required is None:
        num_time_units_required = 104 if time_unit == "week" else 24
    
    # If unspecified, require 5000 units sold per week on average
    month_factor = 1 if time_unit == "week" else 4
    if avg_units_required is None:
        avg_units_required = 5000*month_factor #scale because there are 4 weeks in a month
    
    df = df[[industry_dict["time_var"], "units"]].groupby(industry_dict["time_var"]).sum().reset_index()

    num_time_units_sold = len(df[industry_dict["time_var"]].drop_duplicates())
    avg_units_sold = int(df["units"].mean())
    
    
    # Calculate 'seasonality score', the rsquare value of a simple model where we regress unit sales on seasonality
    try:
        period = 52 if time_unit == "week" else 12
        seasonality = seasonal_decompose(df["units"], period=period, extrapolate_trend=0)  # extrapolate_trend must be 0
        X = pd.DataFrame(seasonality.seasonal)
        X.insert(0, 'const', [1.0]*len(X))

        model = sm.OLS(df["units"], X).fit()
        seas_score = model.rsquared
    # Exceptions imply the data are not seasonal or lack enough time periods (2 years)
    except Exception as e:
        #print(e) #TO DEBUG, UNCOMMENT THIS LINE
        seas_score = 0
    
    
    # First, document which feasibility criteria, if any, are not met.
    feasible = True
    reason_string = ""
    if num_time_units_sold < num_time_units_required:
        feasible = False
        reason_string = "Number of {}s sold is {}, less than the required {} {}s".format(time_unit, num_time_units_sold, num_time_units_required, time_unit)
    
    if avg_units_sold < avg_units_required:
        feasible = False
        this_string = "vg units sold per {} is {}, less than the required {} per {}".format(time_unit, avg_units_sold, avg_units_required, time_unit)
        if reason_string:
            reason_string = reason_string + " & a" + this_string
        else:
            reason_string = "A" + this_string
    
    if num_time_units_sold >= num_time_units_required and seas_score < seas_score_required:
        feasible = False
        this_string = "he seasonality score of {:.3f} is less than the required {}".format(seas_score, seas_score_required)
        if reason_string:
            reason_string = reason_string + " & t" + this_string
        else:
            reason_string = "T" + this_string
    
    # If all criteria were passed, document as such.
    if feasible:
        reason_string = "Feasibility criteria met ({}s sold >= {}, units/{} >= {}, seasonality score >= {})".format(time_unit, num_time_units_required, time_unit, avg_units_required, seas_score_required)
    # We'll pass a segment that has lower-than-accepted units/week but high seasonality
    elif avg_units_sold >= 2000*month_factor and seas_score >= 0.70:
        feasible = True
        reason_string = "High seasonality score (>= 0.70)"
    
    
    return feasible, reason_string, num_time_units_sold, avg_units_sold, seas_score

def rollup(df, groups_list):
    """ Creates groups by replacing specific values in particular columns subject to values in other columns. Takes a list of strings as input.
        Usage is 'in <column> replace (<val1>, <val2>) with <new_val> where <other_column> is <val> and <another_column> is another <val>'
        There can be many values to replace and many filters, but column and new val will have one value for each input string.
        Accepts spark or pandas dataframes.
    """
    
    # Check whether we have a pandas or pyspark dataframe.
    
    pandas = isinstance(df, pd.DataFrame)
    
    for group in groups_list:
        
        # Create a dictionary for the filters, if there are any.
        filt_dict = {}
        if "where" in group:
            group, filts = group.split(" where ")
            filters = filts.split(" and ")
            for filter in filters:
                col, val = filter.split(" is ")
                filt_dict[col.strip("' ")] = val.strip("' ")
        
        # Split the first part of the string into the column specification and the replacement specification.
        try:
            col, repl = group.split(" replace ")
        except:
            raise ValueError("Invalid grouping input: '{}'. Specify the column in which the grouping should occur.".format(group))
        col = col.split(" ")
        if len(col) == 2 and col[0] == "in":
            column = col[1].strip("' ")
        else:
            raise ValueError("Invalid grouping input: '{}'. Must begin with 'in column_name'".format(col))
        
        # Extract the replacement values and store them in a dict.
        try:
            old_vals, new_val = [phrase.strip("' ") for phrase in repl.split(" with ")]
        except:
            raise ValueError("Invalid grouping input: '{}'. Format should be ''('val1','val2') with 'new_val''".format(repl))

        if old_vals[0] == '(' and old_vals[-1] == ')':
            old_vals = [word.strip("', '") for word in old_vals.strip("()").split(",")]
        else:
            raise ValueError("Invalid grouping input: '{}'. Format should be ''('val1','val2')".format(old_vals))

        if pandas:
            # Get the column to operate on.
            newcol = list(df[column])

            # Create a dict of lists from the columns we are filtering with.
            ref_dict = {refcol: list(df[refcol]) for refcol in filt_dict.keys()}

            # Perform the replacements.
            for i, val in enumerate(newcol):
                if val in old_vals: 
                    if all([ref_dict[refcol][i]==filt_dict[refcol] for refcol in ref_dict.keys()]):
                        newcol[i] = new_val

            # Overwrite the column in the dataframe.
            df[column] = newcol
            
        else: #dataframe is a spark df
            
            # A function that checks the conditions on a row and makes a substitution accordingly
            def spark_repl(row):
                row_dict = row.asDict()
                if row_dict[column] in old_vals:
                    if all([row_dict[refcol]==filt_dict[refcol] for refcol in filt_dict.keys()]):
                        row_dict[column] = new_val
                return Row(**row_dict)
            
            # Create an rdd, perform the mapping on all rows
            df_rdd = df.rdd
            df = df_rdd.map(lambda row: spark_repl(row)).toDF()
        
        
    return df

def analyze_feas(df, industry_dict):
    ### pandas implementation
    
    releasable, releasable_reason_string = is_releasable(
        df,
        num_outlets_required=industry_dict["releasability_criteria"][0],
        largest_share_allowed=industry_dict["releasability_criteria"][1]
    )

    feasible, feasible_reason_string, num_time_units_sold, avg_units_sold, seas_score = is_feasible(
        df,
        industry_dict,
        num_time_units_required=industry_dict["feasibility_criteria"][0],
        avg_units_required=industry_dict["feasibility_criteria"][1],
        seas_score_required=industry_dict["feasibility_criteria"][2]
    )

    forecast_decision = "Forecast Feasible" if feasible and releasable else "Forecast likely NOT feasible. Consult with Advanced Insights."
    time_unit = "Week" if industry_dict["time_var"] == "ppweek" else "Month"
    
    # Since we've filtered to a dataframe with only 1 distinct value for each classifier & attribute column, we can just grab the first value with .iloc[0]
    # We put these into a dictionary that we can unpack with ** into the output_data dictionary
    clfs_dict = {clf_name: df[clf_name].iloc[0] for clf_name in industry_dict["clf_cols"][1::2]}
    attrs_dict = {}
    if industry_dict["attribute_level"]:
        attrs_dict["attr_desc"] = df["attr_desc"].iloc[0]
        attrs_dict["attrvl_desc"] = df["attrvl_desc"].iloc[0]

    # Append this segment's data to the main industry results dataframe
    output_data = {"Library": [industry_dict["library"]],
                   **clfs_dict, 
                   **attrs_dict,
                   "Number of {}s Sold".format(time_unit): [num_time_units_sold], 
                   "Average Units Per {}".format(time_unit): [avg_units_sold],
                   "Seasonality Score": [round(seas_score, 4)],
                   "Feasible": [feasible],
                   "Releasable": [releasable],
                   "Feasible Reason": [feasible_reason_string],
                   "Releasable Reason": [releasable_reason_string],
                   "Decision": [forecast_decision]
    }
    
    return pd.DataFrame.from_dict(output_data, orient="columns")

def run(industry_dict, write_csv=False):
    """ Loops through elements of industry_dict and returns a dataframe of feasibility assessments """
    
    industry_dict = autofill_config(industry_dict)
    check_config_values(industry_dict)
    
    print("Updated dictionary values:")
    for key, value in industry_dict.items():
        print("\t", key, ":", value)
    
    start = time.time()
    
    dao = DAO(meta_dictionary=[{"lib": industry_dict["library"], "ref": "rawdata"}],
        env="prod",
        scope="kv-sd-us-secret-scope"
    )
    raw_sales_df = extract_sales_data(dao, industry_dict)
    item_dims_df = extract_item_data(dao, raw_sales_df, industry_dict)

    # We do the initial data prep (loading/merging/aggregating) in Spark because it was too large for Pandas
    df = prep_data(industry_dict, raw_sales_df, item_dims_df)

    #switch to pandas from pyspark
    df = df.toPandas()
        
    # Perform segment groupings (a.k.a. "roll up")
    if "groups" in industry_dict:
        if isinstance(industry_dict["groups"], str):
            industry_dict["groups"] = [industry_dict["groups"]]
        df = rollup(df, industry_dict["groups"]) # Note that we're passing a pandas dataframe here.
    
    # Rather than iterating over all levels of the classifier with nested for loops, 
    # we get all unique combinations of classifiers and attributes/attribute values.
    # Then we filter the dataframe to each of these unique combinations with query() to assess feasibilty for that segment.
    filters_df = df[[*industry_dict["clf_cols"][1::2],*industry_dict["attr_cols"]]].drop_duplicates().reset_index(drop=True)
        #industry_dict["clf_cols"][1::2] yields just classifier columns that end in "_name"
    cols = filters_df.columns
    result_df = pd.DataFrame()
    for i, filter in filters_df.iterrows():
        print("                  Calculating feasiblity and releasability for all segments...{}/{}".format(i, len(filters_df)), end='\r')
        args = [f"""{col}=='{val}'""" for col, val in zip(cols, filter.tolist())]
        argstr = " and ".join(args)
        sub_df = df.query(argstr)
        
        output_data_df = analyze_feas(sub_df, industry_dict)

        result_df = pd.concat([result_df, output_data_df])
    
    ###this is a spark implementation of the above logic in case I change everything to spark later
    #filters_df = df.select(*industry_dict["clf_cols"][::2],*industry_dict["attr_cols"]).distinct()
    #cols = filters_df.columns
    #filters = filters_df.collect()
    #result_df = pd.DataFrame()
    #for i, filter in enumerate(filters):
    #    print("Calculating feasiblity and releasability for all segments...\t{}/{}".format(i, len(filters)), end="\r")
    #    args = [f"""{col}=='{val}'""" for col, val in zip(cols, filter)]
    #    argstr = " and ".join(args)
    #    sub_df = df.filter(argstr)
    #    
    #    output_data_df = analyze_feas(sub_df, industry_dict)
    #
    #    result_df = pd.concat([result_df, output_data_df])
    
    print("Calculating feasiblity and releasability for all segments...Done.                                        ")         
    end = time.time()
    time_elapsed = time.gmtime(end - start)
    print("Job '{}' finished successfully ({}h-{}m-{}s elapsed).".format(industry_dict["job_id"], time_elapsed.tm_hour, time_elapsed.tm_min, time_elapsed.tm_sec))
        
    return result_df

def pull_forecast_data(industry_dict):
    """ If the request is feasible, you can use this to pull the data needed to input into the forecasting model.
        Output has columns: [ppweek, WeekEnd, Month, Quarter, Year, highlevel_name, forecast_name, industry, country, units, asp]
        If data is monthly, then [ppweek, WeekEnd] are replaced with just ppmonth.
    """
    
    industry_dict = autofill_config(industry_dict)
    check_config_values(industry_dict)

    if "country" not in industry_dict:
        raise Exception("Invalid input. Please specify a value for 'country'.")
        
    if "industry" not in industry_dict:
        industry_dict["industry"] = industry_dict["library"][:3].lower()

    print("Updated dictionary values:")
    for key, value in industry_dict.items():
        print("\t", key, ":", value)

    dao = DAO(meta_dictionary=[{"lib": industry_dict["library"], "ref": "rawdata"}],
        env="prod",
        scope="kv-sd-us-secret-scope"
    )

    start = time.time()

    # Extract the data.
    raw_sales_df = extract_sales_data(dao, industry_dict)
    item_dims_df = extract_item_data(dao, raw_sales_df, industry_dict)

    # Join the dataframes and aggregate dollars and units.
    merged_df = raw_sales_df.join(item_dims_df, how='inner', on=['itemid', *industry_dict["classifier"]])
    merged_df = merged_df.groupby(industry_dict["time_var"], *industry_dict["clf_cols"], *industry_dict["attr_cols"])\
                  .agg(func.sum('dollars').alias('dollars'), func.sum('units').alias('units'))

    # Drop coded classifiers since we have the names.
    df = merged_df.drop(*industry_dict["classifier"])
    
    # Perform groupings.
    if "groups" in industry_dict:
        if isinstance(industry_dict["groups"], str):
            industry_dict["groups"] = [industry_dict["groups"]]
        df = rollup(df, industry_dict["groups"]) # Note that we're passing a spark dataframe here.
    
    # Add columns.
    df = df.withColumn('asp', df['dollars'] / df['units'] )
    df = df.select(industry_dict["time_var"], 
                   func.concat_ws("/", *industry_dict["clf_cols"][1::2], *industry_dict["attr_cols"]).alias('forecast_name'),
                   func.lit(industry_dict["industry"]).alias('industry'),
                   func.lit(industry_dict["country"]).alias('country'),
                   'units',
                   'asp')
    
    df = df.withColumn('highlevel_name', df['forecast_name'])
    
    weekend = ['WeekEnd'] if industry_dict["time_var"] == 'ppweek' else []
    
    time_var = industry_dict["time_var"]
    time_vars_str = ",".join(map(str,list(df.select(time_var).distinct().toPandas()[time_var])))
    
    # If the datelookup data is stored in a different location, this variable will need to be changed.
    datelookup = "shared_data.datelookup"
    
    # Retrieve datelookup data.
    query = f"""select distinct {time_var.upper()}, {weekend[0]+"," if weekend else ""} Month, Quarter, Year
            from {datelookup}
            where {time_var.upper()} in ({time_vars_str})  
    """
    
    # Get the spark instance with which to make a query.
    spark = SparkSession.builder.getOrCreate()
    date_df = spark.sql(query).withColumnRenamed(time_var.upper(), time_var)
    
    # Join date data and reorder columns.
    df = df.join(date_df, time_var, 'inner').select(
        industry_dict["time_var"], 
        *weekend,
        'Month',
        'Quarter',
        'Year',
        'highlevel_name',
        'forecast_name',
        'industry',
        'country',
        'units',
        'asp'
    )
                
    end = time.time()
    time_elapsed = time.gmtime(end - start)
    print("Job '{}' finished successfully ({}h-{}m-{}s elapsed).".format(industry_dict["job_id"], time_elapsed.tm_hour, time_elapsed.tm_min, time_elapsed.tm_sec)) 
       
    return df
    

def process_industry(industry_dict):
    """DEPRECATED: Use run(industry_dict) instead. """
    
    print("Function process_industry() is deprecated. Please use run(industry_dict) instead.")
    
    return
    

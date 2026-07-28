cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(".")

# Load the required packages
import TulipaEnergyModel as TEM
import TulipaIO
import TulipaClustering
import DuckDB
import Gurobi
import Distances
import CSV
using DataFrames
using Random
using DuckDB: DBInterface, DuckDB
using Plots

user_input_dir = "obz_data_raw/"

connection = DBInterface.connect(DuckDB.DB, "obz.db")

TulipaIO.read_csv_folder(
    connection,
    user_input_dir,
    replace_if_exists=true,
)
DuckDB.query(connection,
    "ALTER TABLE profiles_wide
        ADD scenario Int32;",
)
DuckDB.query(
    connection,
    "
    UPDATE profiles_wide
    SET scenario = 0;
"
)
TulipaClustering.transform_wide_to_long!(connection, "profiles_wide", "pivot_profiles"; exclude_columns=["year", "timestep", "scenario"])
DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE profiles AS
    FROM pivot_profiles
    ORDER BY profile_name, year, timestep
    "
)
# I have added investment in batteries and electrolyzers
DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE asset AS
    SELECT
        name AS asset,
        type,
        capacity,
        capacity_storage_energy,
        is_seasonal,
        CASE
            WHEN LOWER(name) LIKE '%wind_onshore%'
              OR LOWER(name) LIKE '%wind_offshore%'
              OR LOWER(name) LIKE '%solar%'
              OR LOWER(name) LIKE '%coal%'
              OR LOWER(name) LIKE '%ocgt%'
              OR LOWER(name) LIKE '%gas%'
              OR LOWER(name) LIKE '%nuclear%'
              OR LOWER(name) LIKE '%battery%'
              OR LOWER(name) LIKE '%electrolyzer%'
            THEN 'simple'
            ELSE 'none'
        END AS investment_method,
        CASE 
            WHEN LOWER(name) LIKE '%battery%'
            THEN 'true'
            ELSE 'false'
        END AS storage_method_energy,
        false AS investment_integer
    FROM (
        FROM assets_consumer_basic_data
        UNION BY NAME
        FROM assets_conversion_basic_data
        UNION BY NAME
        FROM assets_producer_basic_data
        UNION BY NAME
        FROM assets_storage_basic_data
    )
    ORDER BY asset
    ",
)

DuckDB.query(
    connection,
    "DELETE FROM asset
WHERE LOWER(asset) LIKE '%ens%'
   OR LOWER(asset) LIKE '%smr%';"
)

DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE t_asset_yearly AS
    FROM (
        FROM assets_consumer_yearly_data
        UNION BY NAME
        FROM assets_conversion_yearly_data
        UNION BY NAME
        FROM assets_producer_yearly_data
        UNION BY NAME
        FROM assets_storage_yearly_data
    )
    ",
)
# investment_costs
# wind_onshore = 77356.32865703155
# wind_offshore = 119732.61777406993
# solar = 34342.98027538492
# battery = 77577.8503057667
# for capacity --> battery = 77577.8503057667 * 4 = 310311.401223067
# coal = 420000.0
# ocgt = 55000.0
# gas = 95000.0     
# nuclear = 950000.0
# electrolyzer 800000.0

DuckDB.query(
    connection,
    "
    CREATE OR REPLACE TABLE asset_commission AS
    SELECT
        tay.name AS asset,
        tay.year AS commission_year,
        CASE
            WHEN LOWER(tay.name) LIKE '%wind_onshore%'  THEN 77356.32865703155
            WHEN LOWER(tay.name) LIKE '%wind_offshore%' THEN 119732.61777406993
            WHEN LOWER(tay.name) LIKE '%solar%'          THEN 34342.98027538492
            WHEN LOWER(tay.name) LIKE '%coal%'           THEN 420000.0
            WHEN LOWER(tay.name) LIKE '%ocgt%'            THEN 55000.0
            WHEN LOWER(tay.name) LIKE '%gas%'             THEN 95000.0
            WHEN LOWER(tay.name) LIKE '%nuclear%'         THEN 950000.0
            WHEN LOWER(tay.name) LIKE '%battery%'         THEN 77577.8503057667
            WHEN LOWER(tay.name) LIKE '%electrolyzer%'    THEN 800000.0
            ELSE NULL
        END AS investment_cost,
        CASE 
        WHEN LOWER(tay.name) LIKE '%pl%' THEN 3 * a.capacity
        ELSE 2 * a.capacity END AS investment_limit,
        CASE
            WHEN LOWER(tay.name) LIKE '%battery%' THEN 310311.401223067
            ELSE 0.0
        END AS investment_cost_storage_energy,
        CASE 
            WHEN LOWER(tay.name) LIKE '%battery%' THEN 2 * a.capacity 
            ELSE 0.0
        END AS investment_limit_storage_energy
    FROM t_asset_yearly tay
    JOIN asset a
    ON a.asset = tay.name
    ORDER BY asset;
    "
)
DuckDB.query(
    connection,
    "DELETE FROM asset_commission
WHERE LOWER(asset) LIKE '%ens%'
   OR LOWER(asset) LIKE '%smr%';"
)

DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE asset_milestone AS
    SELECT
        name AS asset,
        year AS milestone_year,
        peak_demand,
        initial_storage_level,
        storage_inflows,
        CASE
            WHEN LOWER(name) LIKE '%wind_onshore%'
              OR LOWER(name) LIKE '%wind_offshore%'
              OR LOWER(name) LIKE '%solar%'
              OR LOWER(name) LIKE '%coal%'
              OR LOWER(name) LIKE '%ocgt%'
              OR LOWER(name) LIKE '%gas%'
              OR LOWER(name) LIKE '%nuclear%'
              OR LOWER(name) LIKE '%battery%'
              OR LOWER(name) LIKE '%electrolyzer%'
            THEN true
            ELSE false
        END AS investable,
    FROM t_asset_yearly
    ORDER by asset
    "
)
DuckDB.query(
    connection,
    "DELETE FROM asset_milestone
WHERE LOWER(asset) LIKE '%ens%'
   OR LOWER(asset) LIKE '%smr%';"
)


DuckDB.query(
    connection,
    "
    CREATE OR REPLACE TABLE asset_both AS
    SELECT
        tay.name AS asset,
        tay.year AS milestone_year,
        tay.year AS commission_year, -- same year, different semantic meaning
        CASE
            WHEN a.investment_method = 'simple' THEN 0
            ELSE tay.initial_units
        END AS initial_units,
        tay.initial_storage_units
    FROM t_asset_yearly tay
    JOIN asset a
    ON a.asset = tay.name
    ORDER BY asset;
    "
)
DuckDB.query(
    connection,
    "DELETE FROM asset_both
WHERE LOWER(asset) LIKE '%ens%'
   OR LOWER(asset) LIKE '%smr%';"
)

# flows
DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE flow AS
    SELECT
        from_asset,
        to_asset,
        carrier,
        capacity,
        is_transport,
    FROM (
        FROM flows_assets_connections_basic_data
        UNION BY NAME
        FROM flows_transport_assets_basic_data
    )
    ORDER BY from_asset, to_asset
    ",
)
DuckDB.query(
    connection,
    "DELETE FROM flow
WHERE LOWER(from_asset) LIKE '%ens%'
   OR LOWER(from_asset) LIKE '%smr%';"
)


DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE t_flow_yearly AS
    FROM (
        FROM flows_assets_connections_yearly_data
        UNION BY NAME
        FROM flows_transport_assets_yearly_data
    )
    ",
)

DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE flow_commission AS
    SELECT
        from_asset,
        to_asset,
        year AS commission_year,
        efficiency AS producer_efficiency,
    FROM t_flow_yearly
    ORDER by from_asset, to_asset
    "
)
DuckDB.query(
    connection,
    "DELETE FROM flow_commission
WHERE LOWER(from_asset) LIKE '%ens%'
   OR LOWER(from_asset) LIKE '%smr%';"
)

DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE flow_milestone AS
    SELECT
        from_asset,
        to_asset,
        year AS milestone_year,
        variable_cost AS operational_cost
    FROM t_flow_yearly
    ORDER by from_asset, to_asset
    "
)
DuckDB.query(
    connection,
    "DELETE FROM flow_milestone
WHERE LOWER(from_asset) LIKE '%ens%'
   OR LOWER(from_asset) LIKE '%smr%';"
)

DuckDB.query(
    connection,
    "CREATE OR REPLACE TABLE flow_both AS
    SELECT
        t_flow_yearly.from_asset,
        t_flow_yearly.to_asset,
        t_flow_yearly.year AS milestone_year,
        t_flow_yearly.year AS commission_year,
        t_flow_yearly.initial_export_units,
        t_flow_yearly.initial_import_units,
    FROM t_flow_yearly
    LEFT JOIN flow
      ON flow.from_asset = t_flow_yearly.from_asset
      AND flow.to_asset = t_flow_yearly.to_asset
    WHERE flow.is_transport = TRUE -- flow_both must only contain transport flows
    ORDER by t_flow_yearly.from_asset, t_flow_yearly.to_asset
    "
)

# save the tables in csv to be reread everytime

output_dir = joinpath(@__DIR__, "input_tables")
mkpath(output_dir)

tables_df = DataFrame(DuckDB.query(
    connection,
    "SELECT table_name
     FROM information_schema.tables
     WHERE table_schema = 'main'"
))

table_names = tables_df.table_name
tables_to_keep = ["profiles", "asset", "asset_both", "asset_commission", "asset_milestone", "assets_profiles", "assets_rep_periods_partitions", "flows_rep_periods_partitions", "assets_timeframe_profiles", "flow", "flow_both", "flow_commission", "flow_milestone", "year_data", "profiles_timeframe", "profiles_wide"]
for t in table_names
    if t in tables_to_keep
        outfile = joinpath(output_dir, "$(t).csv")
        DuckDB.query(
            connection,
            "COPY $t TO '$outfile' (HEADER, DELIMITER ',');"
        )
    end
end


close(connection)
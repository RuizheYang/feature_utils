import os
from pyhive import hive
import pandas as pd
import shutil
import string
import random
from functools import reduce
from nice_feat.raw_data import RawData
import json

HIVE_URL="ec2-54-254-163-134.ap-southeast-1.compute.amazonaws.com"
HIVE_HOST="ec2-54-254-163-134.ap-southeast-1.compute.amazonaws.com"
HIVE_SCHEMA="rdp_ods"
HIVE_PORT=10000
HIVE_USER="hadoop"
HIVE_PASSWORD="Hadoop@2023"

TEMPLATE = """
create table tmp.{table_name}
stored as parquet
as
{sql}
""".strip()

def query_hive(sql:str, table_name:str = None, persist:bool = False):
    """
    Executes a Hive query and returns the result as a pandas DataFrame.
    
    Parameters:
    - sql (str): The Hive query to be executed.
    - table_name (str, optional): The name of the temporary table to be created. If not provided, a random table name will be generated.
    - persist (bool, optional): If True, the temporary table will be persisted in the local file system. If False, the temporary table will be removed after reading the data into a DataFrame.
    
    Returns:
    - df (pandas.DataFrame): The result of the Hive query as a DataFrame.
    
    Raises:
    - Exception: If there is an error executing the Hive query.
    
    Example:
    >>> df = query_hive("select * from rdp_ods.ods_kenya_data_source limit 10")
    """
    
    if table_name is None:
        table_name = "".join(random.choices(string.ascii_lowercase,k = 10))
        
    print("Using table name {}".format(table_name))
    
    success = False
    sql = TEMPLATE.format(sql = sql.strip(), table_name = table_name)
    conn = hive.Connection(host=HIVE_HOST, port=HIVE_PORT, database=HIVE_SCHEMA, username=HIVE_USER)
    cur = conn.cursor()
    
    try:
        cur.execute(sql)
        print("Execute Success")
        success = True
    finally:
        conn.close()
        
    if not success:
        raise Exception("Hive Execution Error")
    
    print("Downloading file from S3")
    os.system(f"aws s3 cp --recursive s3://mib-risk-emr/rdp/hive/tmp/{table_name}/ ./{table_name}/")
    
    
    df = pd.read_parquet(f"./{table_name}/")
    
    if not persist:
        print("Removing local file")
        shutil.rmtree(table_name)
        
    return df

def create_raw_data(df):
    
    ds_names = df['data_source_name'].unique().tolist()

    datasources = {ds: df[df.data_source_name == ds][['sample_id','data']].rename(columns = {'data':ds}) 
                for ds in ds_names}

    datasources = list(datasources.values())

    sideinfo = df[['sample_id','customer_id','sample_time','app_name','country_id']].drop_duplicates(subset = ['sample_id'])
    
    sideinfo['country_id'] = sideinfo['country_id'].astype(str)

    merged_df = reduce(lambda x, y: pd.merge(x, y, on = 'sample_id', how = 'inner'), datasources + [sideinfo])
    
    create_raw_data = lambda row: RawData(app_name = row['app_name'], 
                    country_id = row['country_id'], 
                    sample_time = row['sample_time'],
                    sample_id = row['sample_id'],
                    data = {ds: json.loads(row[ds]) for ds in ds_names}
                    ).dict()
    
    merged_df['raw_data'] = merged_df.apply(create_raw_data, axis = 1)
    
    return merged_df

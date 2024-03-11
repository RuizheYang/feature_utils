import redshift_connector
from sqlalchemy import create_engine
import pandas as pd
import psycopg2  # required for pandas.to_csv(redshift_connection)
from pydantic import BaseModel
import os
from dataclasses import dataclass

class RedshiftConfig(BaseModel):
    user:str
    password:str
    host:str
    database:str
    port:int
    
    @classmethod
    def huan(cls):
        return cls(user = 'wanghuan', 
        port = 5439, 
        password = "miB#huan98167", 
        database = 'prod', 
        host = "redshift-cluster-1-sg.cbfcfzvs8et3.ap-southeast-1.redshift.amazonaws.com")
    
    @property
    def engine_str(self):
        return f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'
        
TIMEOUT_SQL = f"set statement_timeout = {1000 * 60 * 90};"


def execute(sql:str, config = None) -> None:
    if config is None:
        config = RedshiftConfig.huan().dict()
    
    with redshift_connector.connect(**config) as conn:
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(TIMEOUT_SQL)
        cursor.execute(sql)
    

def query(sql:str, config = None) -> pd.DataFrame:
    
    if config is None:
        config = RedshiftConfig.huan().dict()
    
    with redshift_connector.connect(**config) as conn:
        cursor = conn.cursor()
        cursor.execute(TIMEOUT_SQL)
        cursor.execute(sql)
        df = cursor.fetch_dataframe()
    
    return df


MESSAGE_TEMPLATE = """
******************************
  Write to redshift database
  Schema: {schema}
  Table: {table_name}
  Number of Records: {n}
******************************
"""

def write(df:pd.DataFrame, 
          table_name:str, 
          if_exists:str = 'replace', 
          schema:str = 'temp', 
          chunksize:int = 5000, 
          config = None):
    
    if config is None:
        config = RedshiftConfig.huan()
        
    conn = create_engine(config.engine_str)
    n = df.to_sql(table_name, conn, index=False, if_exists=if_exists, schema = schema, chunksize = chunksize, method = 'multi')
    print(MESSAGE_TEMPLATE.format(schema = schema, table_name = table_name, n = n).strip())
    
    


@dataclass
class RedshiftWriter:
    local_fp:str
    table_name:str
    create_sql:str = None
    grant_public:bool = True
    
    def __post_init__(self):
        self.base_name = os.path.basename(self.local_fp)
        self.s3_path = f"s3://mib-risk-emr/rdp/hive/tmp/{self.base_name}"
    
    def save_to_s3(self):
        print("Saving local file to s3...")
        os.system(f"aws s3 cp {self.local_fp} {self.s3_path}")
        
    def create_table(self):
        print("Creating redshift table...")
        execute(self.create_sql)
        
    def grant_to_public(self):
        print("Granting to public...")
        execute(f"grant select on table {self.table_name} to group public")
        
    def copy_data(self):
        print("Copying data to redshift...")
        sql = f"""
        COPY {self.table_name}
        FROM '{self.s3_path}' 
        CREDENTIALS  'aws_access_key_id=AKIAUK4454VMDMW4IMUS;aws_secret_access_key=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'
        region 'ap-southeast-1'
        delimiter ','
        ignoreheader as 1 
        """
        execute(sql)
        print("Success copy data to redshift")
        
    def create(self):
        self.save_to_s3()
        self.create_table()
        if self.grant_public:
            self.grant_to_public()
        self.copy_data()
        
    def add(self):
        self.save_to_s3()
        self.copy_data()
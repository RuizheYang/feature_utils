from pydantic import BaseModel, Field
from typing import Optional
import datetime
import json
from functools import lru_cache
from pathlib import Path
import pkg_resources

now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class Data(BaseModel):
    user_info:Optional[dict] = Field(default_factory=dict)
    amount:Optional[list] = Field(default_factory=list)
    app_list:Optional[list] = Field(default_factory=list)
    contact_list:Optional[list] = Field(default_factory=list)
    device_ref_cust:Optional[list] = Field(default_factory=list)
    device_ref_loan:Optional[list] = Field(default_factory=list)
    level:Optional[dict] = Field(default_factory=dict)
    loan_hist:Optional[list] = Field(default_factory=list)
    mati:Optional[dict] = Field(default_factory=dict)
    nation_ref_loan:Optional[list] = Field(default_factory=list)
    transunion:Optional[dict] =Field(default_factory=dict)
    unnax:Optional[dict] = Field(default_factory=dict)
    unnax_aml:Optional[dict] = Field(default_factory=dict)
    nation_ref_cust:Optional[list] = Field(default_factory=list)
    contact_loan:Optional[list] = Field(default_factory=list)
    contact_customer:Optional[list] = Field(default_factory=list)
    msg_list:Optional[list] = Field(default_factory=list)
    device_info:Optional[dict] = Field(default_factory=dict)
    single_loan_hist:Optional[list] = Field(default_factory=list)
    advance_ocr:Optional[dict] = Field(default_factory=dict)
    advance_face:Optional[dict] = Field(default_factory=dict)
    nubarium_curp:Optional[dict] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
    
class RawData(BaseModel):
    
    app_name:Optional[str] = "example"
    country_id:Optional[str] = "52"
    data:Data = Field(default_factory=Data)
    mobile:Optional[str] = "1234567890"
    sample_id:Optional[str] = '2345667899' 
    sample_time:Optional[str] = now
    type:Optional[str] = "Loan"
    
    
    @classmethod
    @lru_cache()
    def from_template(cls):
        # template_path = Path(__file__).parent / "raw_data_template.json"
        template_path = pkg_resources.resource_filename(__name__, "raw_data_template.json")
        print("Reading Template from file...")
        with open(template_path, 'r') as f:
            template = json.load(f)
        return cls(**template)
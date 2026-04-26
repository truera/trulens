import json
from dataclasses import dataclass,field
from typing import Any,Dict,List,Optional,Union
@dataclass
class ValidationResult:
    is_valid:bool; score:float; errors:List[str]=field(default_factory=list); details:Dict[str,Any]=field(default_factory=dict)
class JsonOutputValidator:
    def __init__(self,rk=None): self.rk=rk or []
    def validate(self,output,schema=None):
        e=[]
        if isinstance(output,str):
            try: p=json.loads(output)
            except json.JSONDecodeError as ex: return ValidationResult(False,0,[str(ex)])
        else:p=output
        if not isinstance(p,dict): return ValidationResult(False,0,["Not a dict"])
        for k in self.rk:
            if k not in p: e.append(f"Missing: {k}")
        s=1.0 if not e else max(0,1-len(e)*0.15)
        return ValidationResult(len(e)==0,s,e,{"type":type(p).__name__})
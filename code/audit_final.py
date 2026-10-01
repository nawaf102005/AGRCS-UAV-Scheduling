"""Verify final source reproduces representative archived experiments (excluding timing)."""
import json,tempfile,hashlib
from pathlib import Path
import numpy as np
from agrcs.model import Config
from run_experiments import run

root=Path(__file__).resolve().parents[1];res=root/'results'
manifest=json.loads((res/'manifest.json').read_text())['cases'];checks=[]
with tempfile.TemporaryDirectory() as td:
    out=Path(td);(out/'traces').mkdir();(out/'calibration').mkdir()
    for case,methods in [('stationary',['A-GRCS','DDQN-Match']),('ack_abrupt',['A-GRCS-ACK']),('frequency4',['A-GRCS'])]:
        args=manifest[case]['config'];args['degrees']=tuple(args['degrees']);args['powers']=tuple(args['powers'])
        c=Config(**args)
        run(c,0,methods,out,case,root/'models')
        for m in methods:
            new=np.load(out/'traces'/f'{case}_0_{m}.npz')
            old=np.load(res/manifest[case]['path']/'traces'/f'{case}_0_{m}.npz')
            gaps={k:float(np.max(abs(new[k]-old[k]))) for k in ['goodput','failure','rf','queue','degree','delivered','alpha_state']}
            assert max(gaps.values())<1e-10,(case,m,gaps)
            checks.append(dict(case=case,method=m,maximum_difference=max(gaps.values())))
audit=dict(reproduction_checks=checks,source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted((root/'code').rglob('*.py'))})
(root/'docs'/'final_source_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(checks,indent=2))

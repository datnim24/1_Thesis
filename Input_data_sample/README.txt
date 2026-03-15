Simple starter CSV pack for the MILP/Flask prototype.

Files:
- roasters.csv
- skus.csv
- jobs.csv
- shift_parameters.csv
- planned_downtime.csv
- manual_disruptions_template.csv
- solver_config.csv

Intended use:
1. Read all CSVs from ./input_data
2. Build baseline deterministic MILP from these files
3. Let the GUI inject extra disruption events using the same schema as manual_disruptions_template.csv

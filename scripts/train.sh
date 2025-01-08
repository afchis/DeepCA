export POLARS_ALLOW_FORKING_THREAD=1
python -m src.train --params admm_true.json
python -m src.train --params admm_false.json

# Prosit GRPC Client as a poetry package

## How to get use the package in my own project using poetry

Add the following line to your `pyproject.toml`.

```
prosit-grpc-client = {git = "git@gitlab.lrz.de:proteomics/students/danielaandrade/prosit-client.git", branch = "poetry_package"}
```


## Examples of how to use the GRPC Client from command line

Example of Prosit predictions generated from a maxquant msms.txt
```
python main.py  --input_path "/media/kusterlab/users_files/DanielaAndrade/1_datasets/Hela_internal/raw/combined/txt"/msms.txt \
                --ce 0.33 \
                --prosit_outdir "/media/kusterlab/users_files/Ludwig_Lautenbacher/prosit_out" \
                --concurrency 1 \
                --server "131.159.152.7:8500" \
                --add_scan_nums \
                --add_ion_masses \
                --batch_size 6000 \
                --add_scan_ids \
                --source "maxquant"
```


Example of Prosit predictions generated from a .csv.
```
python main.py  --input_path "/media/kusterlab/users_files/DanielaAndrade/1_datasets/Hela_internal/novor_out/191204_Hela_R2.csv" \
                --ce 0.33 \
                --prosit_outdir "/media/kusterlab/users_files/Ludwig_Lautenbacher/prosit_out" \
                --concurrency 1 \
                --server "131.159.152.7:8500" \
                --add_scan_nums \
                --add_ion_masses \
                --batch_size 6000 \
                --add_scan_ids \
                --source "novor"
```

```
from predictPROSIT_2 import PredictPROSIT

predictor = PredictPROSIT(server="131.159.152.7:8500",
                          sequences_list=["AAAAAA","AAAAAA"],
                          model_name="proteotypicity"
                          )
# Get predictions
print(predictor.get_predictions())
```

# Army Retention Analysis
Usage:
```
usage: main.py [-h] [--params_file PARAMS_FILE] [--dataroot DATAROOT] [--output OUTPUT] [--simulate]

options:
  -h, --help            show this help message and exit
  --params_file PARAMS_FILE
                        JSON configuration file
  --dataroot DATAROOT   Path to dataset.
  --output OUTPUT       Output path
  --simulate            Simulation
```
**Example**
```
python main.py --params_file configs/config.json --dataroot data/data.csv --output images/img.png --simulate
```

**Note**: Change the configs/config.json for different configuration.


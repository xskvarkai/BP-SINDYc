# BP-SINDYc
```
C:.
│   .gitignore
│   LICENSE
│   README.md
│
├───config
│       koopman_params.yaml
│       simulation_params.yaml
│       sindy_params.yaml
│
├───data
│   ├───processed
│   │       data.json
|   |       pareto_front.png
|   |       real_vs_sim.png
│   │       worker_results.log
│   │
│   └───raw
│           Simulacia.csv
│
└────src
    │   main.py
    │   simulation.py
    │   __init__.py
    │
    ├───data_processing
    │       data_loader.py
    │       data_splitter.py
    │       __init__.py
    │
    ├───models
    │       sindy_model.py
    │       __init__.py
    │
    ├───simulation
    │       dynamic_systems.py
    │       simulator.py
    │       __init__.py
    │
    └───utils
            constants.py
            custom_libraries.py
            helpers.py
            vizualization.py
            __init__.py
```

## Config
Základné nastavenia pre simuláciu a hľadanie SINDY modelu

## src
Spúštenie kódu pre simuláciu je cez simulation.py <br>
Spúštenie kódu pre SINDY model je cez main.py

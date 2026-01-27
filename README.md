# BP-SINDYc
```
C:.
│   .gitignore
│   LICENSE
│   README.md
│
├───.vscode
│       settings.json
│
├───config
│       koopman_params.yaml
│       simulation_params.yaml
│       sindy_params.yaml
│
├───data
│   ├───processed
│   │       data.json
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
            custom_libraries.py
            helpers.py
            vizualization.py
            __init__.py
```

## Config
Zatiaľ neimplementované

## src
Spúštenie kódu pre simuláciu je cez simulation.py <br>
Spúštenie kódu pre SINDY model je cez main.py

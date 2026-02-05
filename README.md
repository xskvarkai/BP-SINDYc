# Bakalárska práca

Tento projekt implementuje komplexný pracovný tok pre odhad modelov Sparse Identification of Nonlinear Dynamical Systems (SINDy), vrátane generovania dát, spracovania, trénovania modelu, hľadania optimálnych konfigurácií a validácie. Projekt je navrhnutý na efektívnosť, využívajúc paralelné spracovanie pre optimalizáciu hľadania modelu.

## Štruktúra Projektu

<pre>
├── README.md
├── config 
│   ├── sindy_params.yaml
│   ├── settings.yaml
│   └── simulation_params.yaml
├── data
│   ├── processed
│   │   └── Aeroshield_with_deriv.csv
│   ├── raw
│   │   ├── Aeroshield.csv
│   │   ├── Aeroshield_val.csv
│   │   └── Simulacia.csv
│   └── reports
│       ├── Aeroshield
│       │   ├── Aeroshield.json
│       │   └── worker_results.log
│       └── Lorenz System
│           ├── data.json
│           ├── pareto_front.png
│           ├── real_vs_sim.png
│           └── worker_results.log
├── requirements.txt
└── src
    ├── __init__.py
    ├── data_ingestion
    │   ├── __init__.py
    │   └── data_loader.py
    ├── data_processing
    │   ├── __init__.py
    │   ├── data_splitter.py
    │   └── sindy_preprocessor.py
    ├── main.py
    ├── models
    │   ├── __init__.py
    │   ├── base.py
    │   └── sindy_estimator.py
    ├── scripts
    │   ├── __init__.py
    │   ├── hardcoded_derivate.py
    │   └── sindy_run_configuration.py
    ├── simulation
    │   ├── __init__.py
    │   ├── dynamic_systems.py
    │   └── simulator.py
    ├── simulation.py
    └── utils
        ├── __init__.py
        ├── config_manager.py
        ├── custom_libraries.py
        ├── helpers.py
        ├── plots.py
        └── sindy_helpers.py
</pre>

## Popis Kľúčových Modulov

### `config/`
Obsahuje konfiguračné súbory vo formáte YAML, ktoré definujú parametre pre načítavanie dát, model SINDy, simulácie a globálne nastavenia aplikácie. `ConfigManager` ([`config_manager.py`](src/utils/config_manager.py)) spravuje načítavanie a prístup k týmto nastaveniam.

### `data_ingestion/data_loader.py`
Trieda `DataLoader` je zodpovedná za načítavanie časovo-radových dát z CSV súborov. Extrahuje stavové premenné (X), riadiace vstupy (U) a určuje časový krok (dt). Podporuje validáciu vstupu, plotovanie dát.

### `data_processing/data_splitter.py`
Trieda `TimeSeriesSplitter` rozdeľuje časovo-radové dáta na trénovacie, validačné a testovacie sady. Ponúka možnosť aplikovať Savitzky-Golay filter pre vyhladenie dát a voliteľné perturbácie tréningovej časti vstupného signálu.

### `data_processing/sindy_preprocessor.py`
Obsahuje funkcie pre predprocesing signálov:
-   `find_noise`: Odhaduje úroveň šumu v signáli pomocou wavelet denoising.
-   `find_periodicity`: Analyzuje periodicitu signálu pomocou Fourierovej transformácie.
-   `estimate_threshold`: Odhaduje rozsah kandidátskych prahov pre riedku regresiu SINDy modelu.
-   `generate_trajectories`: Generuje náhodné pod-trajektórie z trénovacej sady dát.

### `models/sindy_estimator.py`
Trieda `SindyEstimator` je hlavná trieda pre odhad SINDy modelov. Spravuje konfigurácie modelu, vykonáva paralelné hľadanie optimálnych modelov na mriežke parametrov, vyhodnocuje výsledky a podporuje validáciu na testovacích dátach. Využíva `multiprocessing` pre efektívne prehľadávanie konfigurácií.

### `scripts/sindy_run_configuration.py`
Funkcia `run_config` je volaná paralelne v rámci `SindyEstimator`. Pripravuje a vyhodnocuje jeden SINDy model na základe danej konfigurácie a dát. Obsahuje logiku pre kontrolu zložitosti modelu, veľkosti koeficientov a simuláciu pre vyhodnotenie výkonu (R2 score, RMSE, AIC).

### `scripts/hardcoded_derivate.py`
Funkcia `load_and_deriv` načítava dáta, vypočíta časové derivácie pomocou `numpy.gradient` a ukladá ich do nového CSV súboru.

### `simulation/dynamic_systems.py` a `simulation/simulator.py`
Tieto moduly umožňujú simuláciu dynamických systémov definovaných funkciou ODE. `DynamicSystem` ([`dynamic_systems.py`](src/simulation/dynamic_systems.py)) spravuje simulačné parametre, generovanie vstupných signálov a integráciu stavových zmien pomocou metódy Runge-Kutta 4. rádu (`rk4_step` v [`simulator.py`](src/simulation/simulator.py)). Podporuje aj pridávanie šumu do simulovaných dát.

### `utils/custom_libraries.py`
Definuje vlastné knižnice funkcií (`FixedWeakPDELibrary`, `FixedCustomLibrary`), ktoré opravujú možnosti PySINDy. Obsahuje rôzne polynomiálne, goniometrické a kombinačné funkcie použiteľné ako báza pre SINDy.

### `utils/plots.py`
Poskytuje funkcie pre vizualizáciu dát:
-   `plot_trajectory`: Vykresľuje časové trajektórie stavových premenných a vstupných signálov.
-   `plot_pareto`: Vykresľuje Pareto front, ktorý ukazuje kompromis medzi chybou (RMSE) a komplexnosťou modelu.

## Inštalácia a Spustenie

### Inštalácia

1.  **Klonujte repozitár:**
    ```bash
    git clone https://github.com/xskvarkai/BP-SINDYc/
    cd BP-SINDYc/
    ```
2.  **Vytvorte a aktivujte virtuálne prostredie** (odporúčané):
    ```bash
    python -m venv venv
    # Pre Windows
    .\venv\Scripts\activate
    # Pre macOS/Linux
    source venv/bin/activate
    ```
3.  **Nainštalujte závislosti:** 
    ```bash  
    pip install -r requirements.txt  
    ```  
### Spustenie Hlavného Skriptu

Hlavný pracovný tok projektu je definovaný v `main.py`. Tento skript načíta konfigurácie, spracuje dáta, odhadne SINDy modely a vyhodnotí ich.
Paramtre, ktoré sú hľadané treba nastaviť ručne v kóde.

```bash
python src/main.py
```

Pre generovanie simulačných dát dynamického systému môžete spustiť skript `simulation.py`:

```bash
python src/simulation.py
```

Výsledné simulované dáta budú uložené v data/raw/Simulacia.csv (alebo podľa nastavení v `simulation_params.yaml`)


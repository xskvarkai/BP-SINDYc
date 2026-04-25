
## Štruktúra projektu  

Projekt je organizovaný do nasledujúcich hlavných adresárov a súborov:  

```text
.
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
├── src/
│   ├── data_ingestion/
│   │   └── data_loader.py
│   ├── data_processing/
│   │   ├── data_splitter.py
│   │   └── sindy_preprocessor.py
│   ├── models/
│   │   ├── base.py
│   │   ├── koopman_model.py
│   │   ├── sindy_estimator.py
│   │   └── linearized_model.py
│   ├── simulation/
│   │   ├── dynamics_systems.py
│   │   └── simulator.py
│   ├── utils/
│   │   ├── config_manager.py
│   │   ├── custom_libraries.py
│   │   ├── helpers.py
│   │   ├── koopman_helpers.py
│   │   ├── plots.py
│   │   └── sindy_helpers.py
│   ├── main.py
│   ├── main_koopman.py
│   ├── main_linear.py
│   ├── main_sindy.py
│   ├── model_recostruction.py
│   └── run_simulation.py
└── requirements.txt
```  

## Popis hlavných komponentov  

### Konfigurácia (`config/`)  
Adresár obsahuje YAML súbory, ktoré definujú parametre pre rôzne časti projektu:  
- **`koopman_params.yaml`:** Špecifikuje parametre pre načítanie dát, delenie dát a konfiguráciu Koopman modelu.  
- **`settings.yaml`:** Obsahuje globálne nastavenia, cesty k dátam a predvolené parametre pre metódy diferenciácie, optimalizátory a knižnice funkcií pre SINDy.  
- **`simulation_params.yaml`:** Definuje parametre pre simulácie dynamických systémov, vrátane časového kroku a počiatočných podmienok.  
- **`sindy_params.yaml`:** Obsahuje špecifické parametre pre SINDy model, ako sú nastavenia pre načítanie a delenie dát, predspracovanie, parametre pre optimalizáciu a obmedzenia hľadania.  

### Dátové súbory (`data/`)  
Adresár `data` je rozdelený na tri podadresáre:  
- **`raw`:** Surové, neupravené dáta vo formáte CSV (napr. `Aeroshield.csv`, `Floatshield.csv`).  
- **`processed`:** Spracované dáta, ktoré boli upravené a pripravené pre modelovanie (napr. `Aeroshield_with_deriv.csv`, `Floatshield_with_deriv.csv`).  
- **`reports`:** Výstupy analýz, ako sú JSON súbory s výsledkami modelovania a PNG obrázky s vizualizáciami (napr. Pareto front, simulované trajektórie).  

### Zdrojové kódy (`src/`)  
- **`data_ingestion/data_loader.py`:** Trieda `DataLoader` je zodpovedná za načítanie dát z CSV súborov. Poskytuje metódy na extrakciu stavových premenných (X), riadiacich vstupov (U) a určenie časového kroku (dt). Podporuje voliteľné vyhladzovanie dát pomocou Savitzky-Golay filtra a vizualizáciu načítaných dát.  
- **`data_processing/data_splitter.py`:** Trieda `TimeSeriesSplitter` delí časové rady na trénovacie, validačné a testovacie sady. Umožňuje aplikáciu Savitzky-Golay filtra a perturbáciu vstupného signálu.  
- **`data_processing/sindy_preprocessor.py`:** Modul obsahuje funkcie pre predspracovanie dát pred aplikáciou SINDy modelu. Zahŕňa odhad úrovne šumu (`find_noise`) pomocou wavelet transformácie, detekciu periodicity signálu (`find_periodicity`) pomocou Fourierovej transformácie a generovanie subtrajektórií (`generate_trajectories`) z trénovacích dát.  
- **`main_koopman.py`:** Skript pre konfiguráciu, trénovanie a vyhodnocovanie Koopman modelu. Načíta dáta, aplikuje definované pozorovateľné funkcie a regresor (EDMDc), vyhodnotí model a exportuje výsledky.  
- **`main_sindy.py`:** Skript pre rozsiahle trénovanie a vyhodnocovanie SINDy modelu. Konfiguruje rôzne kombinácie knižníc funkcií, optimalizátorov a metód diferenciácie. Vykonáva paralelné hľadanie najlepšej konfigurácie a exportuje výsledky.  
- **`model_recostruction.py`:** Skript sa zameriava na rekonštrukciu SINDy modelu. Nastavuje importy, konfigurácie, načítava a delí dáta, definuje nelineárne funkcie pre SINDy knižnicu a využíva optimalizáciu na identifikáciu riadiacich rovníc.  
- **`models/base.py`:** Základná trieda `BaseSindyEstimator`, ktorá poskytuje rámec pre konfiguráciu SINDy modelov. Umožňuje dynamické nastavenie metód diferenciácie, optimalizátorov a knižníc funkcií, a generuje všetky možné kombinácie konfigurácií.  
- **`models/koopman_model.py`:** Trieda `KoopmanModel` pre modelovanie dynamických systémov pomocou Koopman operátora. Zahŕňa škálovanie dát, trénovanie, vyhodnocovanie výkonu, simuláciu a export modelových parametrov. Obsahuje aj metódy pre interpretáciu vlastných čísel.  
- **`models/sindy_estimator.py`:** Trieda `SindyEstimator` pre odhad SINDy modelov. Spravuje konfigurácie, vykonáva paralelné hľadanie optimálnych modelov s dôrazom na pamäťovú optimalizáciu a vyhodnocuje výsledky. Obsahuje tiež funkcie pre validáciu na testovacích dátach a export dát.  
- **`run_simulation.py`:** Skript, ktorý inicializuje simuláciu dynamického systému, načítava a spracováva dáta. Definuje model systému a simuluje ho.  
- **`scripts/sindy_run_configuration.py`:** Funkcia `run_config` pre beh konfigurácie SINDy modelu. Serializuje konfiguráciu, generuje náhodný „random seed“, predspracuje dáta, zostrojí model a vyhodnotí ho pomocou validačných dát. Vráti metriky výkonu a rovnice modelu.  
- **`simulation/dynamic_systems.py`:** Trieda `DynamicSystem` reprezentuje dynamický systém definovaný diferenciálnou rovnicou. Umožňuje simuláciu trajektórie systému s voliteľným vstupným signálom a počiatočných podmienkach, využívajúc numerickú metódu Runge-Kutta 4. rádu.  
- **`simulation/simulator.py`:** Modul poskytuje funkcie pre numerickú integráciu (`RK4_step`) a generovanie vstupných signálov (`generate_input_signal`) pre dynamický systémy.  
- **`utils/config_manager.py`:** Trieda `ConfigManager` načíta a spravuje konfiguračné nastavenia z YAML súborov. Umožňuje získavať parametre pomocou bodkovej notácie a riešiť cesty k súborom relatívne k koreňovému adresáru projektu.  
- **`utils/custom_libraries.py`:** Definuje triedy `FixedWeakPDELibrary` a `FixedCustomLibrary`, ktoré rozširujú funkcionalitu `pysindy` pre špecifické použitie knižnice funkcií, vrátane polynomiálnych, racionálnych a trigonometrických funkcií.  
- **`utils/helpers.py`:** Všeobecné pomocné funkcie.  
- **`utils/koopman_helpers.py`:** Pomocné funkcie pre Koopman model, ktoré zahŕňajú zostrojenie, simuláciu a vyhodnocovanie modelu.  
- **`utils/plots.py`:** Modul pre vizualizáciu dát a výsledkov. Obsahuje funkcie pre vykresľovanie trajektórie (`plot_trajectory`), Pareto fronty (`plot_pareto`) a Koopman spektra (`plot_koopman_spectrum`).  
- **`utils/sindy_helpers.py`:** Pomocné funkcie pre SINDy model, ktoré slúžia na sanáciu knižníc, zostrojenie a simuláciu modelu, filtrovanie koeficientov a vyhodnocovanie výkonu modelu pomocou RMSE, R2 skóre a AIC.  

## Inštalácia  

Projekt vyžaduje Python 3.x. Závislosti projektu sú uvedené v tabuľke.  

| Knižnica | Verzia |  
| :--- | :--- |  
| cvxopt | 1.3.3 |  
| cvxpy | 1.8.1 |  
| gurobipy | 13.0.1 |  
| matplotlib | 3.10.8 |  
| numpy | 2.4.0 |  
| sympy | 1.14.0 |  
| pandas | 2.3.3 |  
| Pebble | 5.2.0 |  
| pykoopman | 1.2.1 |  
| pysindy | 2.1.0 |  
| PyWavelets | 1.9.0 |  
| PyYAML | 6.0.3 |  
| scikit-learn | 1.8.0 |  
| scikit-optimize | 0.10.2 |  
| scipy | 1.16.3 |  

## Použitie  

Projekt je možné spúšťať pomocou hlavných skriptov umiestnených v adresári `src/`. Konfigurácie pre jednotlivé modely a simulácie sú definované v súboroch `config/*.yaml`.  

### Príklady Spustenia:  
- **Koopman model (`main_koopman.py`)**: Trénuje a vyhodnocuje Koopmanov model.  
  ```bash
  python src/main_koopman.py
  ```  

- **SINDy model (`main_sindy.py`)**: Trénuje a vyhodnocuje SINDy model, vrátane rozsiahleho vyhľadávania parametrov.  
  ```bash
  python src/main_sindy.py
  ```  

- **Rekonštrukcia SINDy modelu (`model_recostruction.py`)**: Ukazuje proces rekonštrukcie SINDy modelu.  
  ```bash
  python src/model_recostruction.py
  ```  

- **Spustenie simulácie (`run_simulation.py`)**: Simuluje dynamický systém na základe definovaných rovníc.  
  ```bash
  python src/run_simulation.py
  ```  

- **Linearizácia modelu (`main_linear.py`)**: Linearizuje dynamický systém na základe definovaných, pracovných bodov.  
  ```bash
  python src/main_linear.py
  ```  

## Konfigurácia  
Všetky konfiguračné parametre sú spravované triedou `ConfigManager` a načítané z YAML súborov v adresári `config/`.  
- `koopman_params.yaml`: Špecifické nastavenia pre Koopmanov model.  
- `sindy_params.yaml`: Špecifické nastavenia pre SINDy model.  
- `simulation_params.yaml`: Parametre pre simulácie dynamických systémov (napr. časový krok, počiatočné podmienky).  
- `settings.yaml`: Všeobecné nastavenia, cesty k dátam, predvolené parametre pre metódy diferenciácie, optimalizátory a knižnice funkcií pre SINDy.  

## Rozšíriteľnosť  
- **Vlastné knižnice funkcií**: Modul `utils/custom_libraries.py` umožňuje definovať vlastné funkcie pre SINDy, čím sa rozširuje priestor hľadaných modelov.  
- **Vizualizácie**: Modul `utils/plots.py` obsahuje funkcie pre generovanie rôznych grafov a vizualizácií, ktoré je možné prispôsobiť alebo rozšíriť.  
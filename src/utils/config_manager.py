import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

class ConfigManager:
    def __init__(self, config_dir: Union[str, Path], default_env: str = "development"):
        """
        Initializes the ConfigManager.
        """

        self.config_dir = Path(config_dir)
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Configuration directory '{self.config_dir}' not found.")
        
        self._config: Dict[str, Any] = {}
        self.default_env = default_env
        self._current_env: Optional[str] = None

        self.project_root = self.config_dir.parent

    def load_config(self, config_name: str, enviroment: Optional[str] = None) -> None:
        """
        Loads a YAML configuration file into the manager's internal configuration dictionary.
        If an environment is specified, it attempts to load environment-specific settings
        and merges them with the base configuration.
        """

        env_to_load = enviroment if enviroment is not None else self.default_env
        base_filepath = self.config_dir / f"{config_name}.yaml"
        env_filepath = self.config_dir / f"{config_name}_{env_to_load}.yaml"

        config_data = {}

        # Nacita zakladnu konfiguraciu
        if base_filepath.is_file():
            with open(base_filepath, "r", encoding="utf-8") as file:
                base_data = yaml.safe_load(file)
                if base_data:
                    config_data.update(base_data)
        
        else:
            # Ak neexistuje zakladna konfiguracia, 
            # ale moze existovat konfiguracia specificka pre dane prostredie
            pass

        # Neexistuje ani jedna konfigaracia, ide o chybu
        if not config_data and not base_filepath.is_file() and not env_filepath.is_file():
            raise FileNotFoundError(
                f"No configuration file found for '{config_name}' "
                f"or '{config_name}_{env_to_load}'. "
                f"Checked: {base_filepath} and {env_filepath}"
            )
        
        # Kluc najvyssej urovne zodpoveda prostrediu
        # pouzijeme jeho nastavenia
        if env_to_load in config_data:
            self._config[config_name] = config_data[env_to_load]
            if base_filepath.is_file(): # Zlucenie zakladneho so specifickym prostredim
                base_without_env_key = {key: value for key, value in config_data.items() 
                                        if key != env_to_load}
                self._deep_merge(self._config[config_name], base_without_env_key)

        else:
            self._config[config_name] = config_data

    def _deep_merge(self, dict1: Dict, dict2: Dict) -> None:
        """Recursively merges second dictionary into first dictionary."""
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                self._deep_merge(dict1[key], value)
            else:
                dict1[key] = value

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration parameter using dot notation (e.g., 'database.host').
        """
        
        parts = key.split('.')
        current_config = self._config

        try:
            for part in parts:
                current_config = current_config[part]
            return current_config
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration parameter '{key}' not found and no default provided.")
        
    def get_path(self, key: str, default: Optional[Union[str, Path]] = None) -> Path:
        """
        Retrieves a file path from the configuration, resolving it relative to the
        project root.
        """

        path_str_or_path = self.get_param(key, default)

        if path_str_or_path is None:
            raise KeyError(f"Path configuration '{key}' not found and no default provided.")
        
        if not isinstance(path_str_or_path, (str, Path)):
            raise TypeError(f"Configuration value for '{key}' must be a string or Path object, got {type(path_str_or_path)}")
        
        return self.project_root / Path(path_str_or_path)
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Returns the entire loaded configuration dictionary.
        """
        return self._config.copy()
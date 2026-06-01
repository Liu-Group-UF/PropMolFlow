"""
Valency table loader with JSON schema validation.

This module provides functionality to load and validate valency tables from JSON files,
replacing hard-coded Python dictionaries for better reproducibility and maintainability.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Union

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    warnings.warn(
        "jsonschema not available. Schema validation will be skipped. "
        "Install with: pip install jsonschema",
        UserWarning
    )


def load_valency_table(table_name: str, validate_schema: bool = True) -> Dict[str, Any]:
    """
    Load and validate a valency table from JSON file.
    
    Args:
        table_name (str): Name of the valency table to load
                         ('tuple', 'legacy', or custom filename)
        validate_schema (bool): Whether to perform JSON schema validation
        
    Returns:
        dict: Loaded and validated valency table
        
    Raises:
        FileNotFoundError: If the valency table file is not found
        jsonschema.ValidationError: If schema validation fails
        ValueError: If the table format is invalid
    """
    # Map table names to files
    table_files = {
        'tuple': 'geom_drugs_h_tuple_valencies.json',
        'legacy': 'legacy_valencies.json'
    }
    
    # Determine file path
    if table_name in table_files:
        filename = table_files[table_name]
    else:
        filename = table_name if table_name.endswith('.json') else f"{table_name}.json"
    
    # Get valency tables directory
    current_dir = Path(__file__).parent
    valency_dir = current_dir.parent / 'valency_tables'
    table_path = valency_dir / filename
    
    if not table_path.exists():
        raise FileNotFoundError(f"Valency table not found: {table_path}")
    
    # Load JSON file
    with open(table_path, 'r') as f:
        data = json.load(f)
    
    # Schema validation
    if validate_schema and HAS_JSONSCHEMA:
        schema_file = _get_schema_file(table_name, data)
        if schema_file:
            schema_path = valency_dir / schema_file
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema = json.load(f)
                try:
                    jsonschema.validate(data, schema)
                except jsonschema.ValidationError as e:
                    raise jsonschema.ValidationError(
                        f"Valency table validation failed for {table_name}: {e.message}"
                    )
    
    # Convert to the format expected by the stability functions
    return _convert_to_internal_format(data, table_name)


def _get_schema_file(table_name: str, data: Dict[str, Any]) -> Union[str, None]:
    """Determine the appropriate schema file for validation."""
    if '$schema' in data:
        return Path(data['$schema']).name
    elif table_name == 'legacy':
        return 'legacy_valency_schema.json'
    elif table_name == 'tuple':
        return 'valency_schema.json'
    return None


def _convert_to_internal_format(data: Dict[str, Any], table_name: str) -> Dict[str, Any]:
    """
    Convert loaded JSON data to internal format expected by stability functions.
    
    Args:
        data (dict): Loaded JSON data
        table_name (str): Name of the table being converted
        
    Returns:
        dict: Converted valency table in internal format
    """
    valency_table = data['valency_table']
    converted = {}
    
    for element, charge_dict in valency_table.items():
        converted[element] = {}
        for charge_str, valencies in charge_dict.items():
            charge_int = int(charge_str)
            
            if table_name == 'tuple':
                # Convert list of lists to list of tuples for tuple tables
                converted[element][charge_int] = [tuple(v) for v in valencies]
            else:
                # Keep as-is for legacy tables
                converted[element][charge_int] = valencies
    
    return converted


def get_available_tables() -> list:
    """
    Get list of available valency tables.
    
    Returns:
        list: Names of available valency tables
    """
    current_dir = Path(__file__).parent
    valency_dir = current_dir.parent / 'valency_tables'
    
    if not valency_dir.exists():
        return []
    
    tables = []
    for json_file in valency_dir.glob('*.json'):
        if not json_file.name.endswith('_schema.json'):
            # Remove .json extension and add to list
            table_name = json_file.stem
            tables.append(table_name)
    
    return sorted(tables)


def validate_table_format(valency_table: Dict[str, Any]) -> bool:
    """
    Validate that a valency table has the expected internal format.
    
    Args:
        valency_table (dict): Valency table to validate
        
    Returns:
        bool: True if format is valid
        
    Raises:
        ValueError: If format is invalid with detailed error message
    """
    if not isinstance(valency_table, dict):
        raise ValueError("Valency table must be a dictionary")
    
    for element, charge_dict in valency_table.items():
        if not isinstance(element, str) or not element.isalpha():
            raise ValueError(f"Invalid element symbol: {element}")
        
        if not isinstance(charge_dict, dict):
            raise ValueError(f"Element {element} must map to a dictionary of charges")
        
        for charge, valencies in charge_dict.items():
            if not isinstance(charge, int):
                raise ValueError(f"Charge must be integer, got {type(charge)} for element {element}")
            
            # Validate valency format (can be int, list, or list of tuples)
            if not isinstance(valencies, (int, list)):
                raise ValueError(f"Invalid valency format for {element}[{charge}]: {type(valencies)}")
    
    return True


# For backward compatibility, provide direct access to loaded tables
def load_geom_drugs_h_tuple_valencies():
    """Load the main tuple-based valency table."""
    return load_valency_table('tuple')


def load_geom_drugs_h_legacy_valencies():
    """Load the legacy valency table."""
    return load_valency_table('legacy')

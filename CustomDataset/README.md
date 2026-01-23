## Using a Custom Dataset in PropMolFlow

To use a custom dataset in PropMolFlow, you need to modify several files:

- **Prepare your dataset files** including an SDF file for molecular structures and a CSV file for molecular properties (please make sure two files are aligned in order)
- **Update the `config.yaml` files** for training the flow matching model and regressor model for property prediction
- **Create a new `process_dataset_cond.py` script** for data processing of your custom dataset
- **Modify `propmolflow/data_processing/dataset.py`** to load your custom dataset
- **Update `propmolflow/property_regressor/test_regressor.py`** to evaluate the regressor model on your custom dataset
- **Update `str2prop_sampler.py`** to define molecular size and property dependencies

## Example: QMe14S Dataset

The [QMe14S dataset](https://doi.org/10.1021/acs.jpclett.5c00839) is used here as an example to demonstrate how to integrate a custom dataset into PropMolFlow. You can download a version of revised QMe14S used by our MolGuidance work on [zenodo](https://zenodo.org/records/16847162). QMe14S contains 186k molecules consisting of 14 elements. In this example, we use a csv file contains single property dipole moment for demonstration. **All configuration files and data processing scripts need to be modified are provided in the `CustomDataset` folder.**

### Dataset Files

Create two directories under `./data`:
- `./data/qme14s_raw`
- `./data/qme14s`

Place your dataset files in the `./data/qme14s_raw` folder:
- `qme14s.sdf` (molecular structures)
- `qme14s.csv` (molecular properties)

**Important:** Ensure that the order of molecules in the SDF file matches the order of entries in the CSV file.

*Note: These files are not provided in the repository due to size limitations. You can download them from the [our new work **MolGuidance**](https://github.com/Liu-Group-UF/MolGuidance) or [original source](https://doi.org/10.1021/acs.jpclett.5c00839).*

### Update Config Files

Create configuration files for your custom dataset:
- `qme14s.yaml` in `./checkpoints` for training the flow model on the QMe14S dataset. Where to modify: 
  - `atom_map:` use the atom types present in your dataset
  - `processed_data_dir:` set to `./data/qme14s` in example case
  - `raw_data_dir:` set to `./data/qme14s_raw` in example case

- `regressor_qme14s.yaml` in `./propmolflow/property_regressor/configs` for training the regressor model on the QMe14S dataset. Where to modify:
  - `dataset_name:` set to `qme14s` in example case
  - `processed_data_dir:` set to `./data/qme14s` in example case
  - `n_atom_types`: set to number of atom types in your dataset, e.g., 14 for QMe14S
  - `output_dir:` set to desired output directory for model checkpoints

- `test_qme14s.yaml` in `./propmolflow/property_regressor/configs` for testing the regressor model on the QMe14S dataset. Where to modify:
  - `atom_map:` use the atom types present in your dataset
  - `processed_data_dir:` set to `./data/qme14s` in example case
  - `dataset_name:` set to `qme14s` in example case

### Data Preparation

Create a new data processing script `process_qme14s_cond.py` to featurize the molecules and compute conditional probabilities. Running this script will generate processed data files in the `./data/qme14s` directory. Where to modify (based on **process_qme14s_cond.py** in the CustomDataset folder):
- line 40: specify the SDF file name
- line 152: specify the property name in the CSV file
- line 280: specify the CSV file name

Place it in parent folder `PropMolFlow` and run it with path to new config file.E.g.
```bash
python process_qme14s_cond.py --config_file checkpoints/qme14s.yaml
```

### Modify Dataset Loader

In `propmolflow/data_processing/dataset.py`, add a new condition to load the QMe14S dataset when the dataset name is `'qme14s'`. Where to modify (based on **dataset.py** in the CustomDataset folder):
- line 159: modify the dataset name based on your custom dataset
- line 166: modify the dataset name based on your custom dataset

Use modified `dataset.py` to replace the original one in `propmolflow/data_processing/`.

### Update Regressor Evaluation

In `propmolflow/property_regressor/test_regressor.py`, add a new condition to evaluate the regressor model on the QMe14S dataset when the dataset name is `'qme14s'`. Where to modify (based on **test_regressor.py** in the CustomDataset folder):
- line 188: modify the dataset name based on your custom dataset

Use modified `test_regressor.py` to replace the original one in `propmolflow/property_regressor/`.

### Update Size Sampler

In `str2prop_sampler.py`, add a new condition to handle the molecular size and property dependencies for the QMe14S dataset when the dataset name is `'qme14s'`. Where to modify (based on **str2prop_sampler.py** in the CustomDataset folder):
- line 52: modify the dataset name based on your custom dataset
- line 53: modify `data/qme14s_raw/train_mol_idxs.npy` based on your custom dataset, this file is generated during data processing
- line 54: modify path to sdf file based on your custom dataset
- line 57: modify path to csv file based on your custom dataset
- line 66: modify property name based on your custom csv file

Use modified `str2prop_sampler.py` to replace the one in parent folder `PropMolFlow`.

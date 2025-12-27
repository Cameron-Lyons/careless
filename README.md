# Careless

A Python package for detecting careless responding in survey data using various statistical indices and methods.

## Overview

When taking online surveys, participants sometimes respond to items without regard to their content. These types of responses, referred to as **careless** or **insufficient effort responding**, constitute significant problems for data quality, leading to distortions in data analysis and hypothesis testing, such as spurious correlations.

The `careless` package provides solutions designed to detect such careless/insufficient effort responses by allowing easy calculation of indices proposed in the literature. For a comprehensive review of these methods, see [Curran (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0022103115000931?via%3Dihub).

## Features

- **Multiple Detection Methods**: Supports various indices for detecting careless responding
- **Flexible Input**: Works with lists, numpy arrays, and pandas DataFrames
- **Robust Implementation**: Handles missing data, edge cases, and provides comprehensive error handling
- **Performance Optimized**: Efficient algorithms for large datasets
- **Comprehensive Documentation**: Detailed docstrings with examples for all functions

## Installation

### From PyPI (when available)
```bash
pip install careless-py
```

### From Source
```bash
git clone https://github.com/Cameron-Lyons/careless-py.git
cd careless-py
pip install -e .
```

### Optional Dependencies
For enhanced functionality (e.g., advanced Mahalanobis distance methods), install with full dependencies:
```bash
pip install careless-py[full]
```

### Using uv (Recommended for Development)

This project uses [uv](https://docs.astral.sh/uv/) for fast, reproducible dependency management. Install uv first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and install:
```bash
git clone https://github.com/Cameron-Lyons/careless-py.git
cd careless-py
uv sync --extra full   # Install with all optional dependencies
```

For development with all dev tools:
```bash
uv sync --extra dev
```

Run commands in the virtual environment:
```bash
uv run pytest          # Run tests
uv run ruff check .    # Run linter
uv run mypy src/       # Run type checker
```

## Quick Start

```python
import numpy as np
from careless import evenodd, irv, longstring, mahad, psychsyn

# Sample survey data (rows = participants, columns = items)
data = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8],  # Participant 1
    [2, 2, 2, 2, 5, 5, 5, 5],  # Participant 2 (suspicious pattern)
    [3, 4, 3, 4, 6, 7, 6, 7],  # Participant 3
])

# Check even-odd consistency
factors = [4, 4]  # Two factors with 4 items each
consistency_scores = evenodd(data, factors)
print("Even-odd consistency scores:", consistency_scores)

# Check intra-individual response variability
irv_scores = irv(data)
print("IRV scores:", irv_scores)

# Check for long strings of identical responses
longest_strings = longstring(data)
print("Longest strings:", longest_strings)
```

## Available Functions

### Consistency Indices

#### `evenodd(x, factors, diag=False)`
Computes the Even-Odd Consistency Index by dividing unidimensional scales using an even-odd split.

**Parameters:**
- `x`: Input data (2D array/list) where rows are individuals and columns are responses
- `factors`: List of integers specifying the length of each factor
- `diag`: Boolean to return diagnostic values (number of valid correlations per individual)

**Returns:**
- Array of even-odd consistency scores (average correlations per individual)
- If `diag=True`: Tuple of (scores, diagnostic_values)

**Example:**
```python
data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
factors = [4, 2]  # First factor has 4 items, second has 2
scores = evenodd(data, factors)
```

#### `psychsyn(x, pairs, method='synonyms', seed=None)`
Computes the Psychometric Synonyms/Antonyms Index based on correlated item pairs.

**Parameters:**
- `x`: Input data (2D array/list)
- `pairs`: List of item pairs [(item1, item2), ...]
- `method`: 'synonyms' or 'antonyms'
- `seed`: Random seed for reproducibility

**Returns:**
- Array of psychometric synonym/antonym scores

**Example:**
```python
data = [[1, 2, 3, 4], [2, 3, 4, 5]]
pairs = [(0, 1), (2, 3)]  # Item pairs to correlate
scores = psychsyn(data, pairs, method='synonyms')
```

#### `psychant(x, pairs, seed=None)`
Convenience wrapper for `psychsyn` that computes psychological antonyms.

### Response Pattern Functions

#### `longstring(x, avg=False)`
Computes the longest (and optionally, average) length of consecutive identical responses.

**Parameters:**
- `x`: Input data (2D array/list)
- `avg`: Boolean to also return average string length

**Returns:**
- Array of longest string lengths per individual
- If `avg=True`: Tuple of (longest_strings, average_strings)

**Example:**
```python
data = [[1, 1, 1, 2, 3], [1, 2, 3, 4, 5]]
longest, avg = longstring(data, avg=True)
```

#### `irv(x, consecutive=None)`
Computes the Intra-individual Response Variability (IRV), the standard deviation of responses across consecutive items.

**Parameters:**
- `x`: Input data (2D array/list)
- `consecutive`: Number of consecutive items to analyze (default: all items)

**Returns:**
- Array of IRV scores per individual

**Example:**
```python
data = [[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]
irv_scores = irv(data)
```

### Statistical Outlier Functions

#### `mahad(x, method='classic', threshold=None, **kwargs)`
Computes Mahalanobis Distance to identify multivariate outliers.

**Parameters:**
- `x`: Input data (2D array/list)
- `method`: Detection method ('classic', 'robust', 'mcd', 'mve')
- `threshold`: Custom threshold for outlier detection
- `**kwargs`: Additional method-specific parameters

**Returns:**
- Array of Mahalanobis distances per individual
- If `threshold` provided: Tuple of (distances, outlier_flags)

**Example:**
```python
data = [[1, 2, 3], [4, 5, 6], [1, 1, 1]]
distances, outliers = mahad(data, method='robust', threshold=0.95)
```

## Advanced Usage

### Working with Different Data Types

The package supports various input formats:

```python
import numpy as np
import pandas as pd

# Numpy arrays
data_np = np.array([[1, 2, 3], [4, 5, 6]])

# Lists
data_list = [[1, 2, 3], [4, 5, 6]]

# Pandas DataFrames
df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
data_df = df.values

# All work the same way
scores = evenodd(data_np, [3])
```

### Handling Missing Data

The functions handle missing data (NaN values) appropriately:

```python
import numpy as np

data_with_nans = np.array([
    [1, 2, np.nan, 4],
    [np.nan, 2, 3, 4],
    [1, 2, 3, 4]
])

# Functions will handle NaN values appropriately
scores = evenodd(data_with_nans, [4])
```

### Custom Thresholds and Parameters

```python
# Custom Mahalanobis distance threshold
distances, outliers = mahad(data, threshold=0.99)

# Custom IRV analysis on consecutive items
irv_scores = irv(data, consecutive=5)

# Psychometric synonyms with custom pairs
pairs = [(0, 1), (2, 3), (4, 5)]
syn_scores = psychsyn(data, pairs, method='synonyms')
```

## Performance Considerations

- **Large Datasets**: For datasets with >10,000 participants, consider processing in chunks
- **Memory Usage**: Functions create copies of input data for processing
- **Parallel Processing**: Consider using multiprocessing for very large datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{careless2024,
  title={Careless: Python package for detecting careless responding},
  author={Lyons, Cameron},
  year={2024},
  url={https://github.com/Cameron-Lyons/careless}
}
```

## References

- Curran, P. G. (2016). Methods for the detection of carelessly invalid responses in survey data. *Journal of Experimental Social Psychology*, 66, 4-19.
- Dunn, A. M., Heggestad, E. D., Shanock, L. R., & Theilgard, N. (2018). Intra-individual response variability as an indicator of insufficient effort responding: Comparison to other indicators and relationships with individual differences. *Journal of Business and Psychology*, 33(1), 105-121.

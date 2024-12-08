# Useful_Python_librairies
# Pandas Functions Reference Table

## Series Creation and Manipulation

| Function | Description | Example |
|----------|-------------|---------|
| `pd.Series()` | Creates a Series from a list or dictionary | `pd.Series([1, 2, 3], index=['a', 'b', 'c'])` |
| `series.name` | Sets name of the Series | `series.name = 'Population'` |
| `series.index.name` | Sets name of the index | `series.index.name = 'Cities'` |
| `series.isnull()` | Checks for null values | `series.isnull()` |
| `series.notnull()` | Checks for non-null values | `series.notnull()` |
| `series.reset_index()` | Resets index to default integers | `series.reset_index(drop=True, inplace=True)` |

## DataFrame Creation and Basic Information

| Function | Description | Example |
|----------|-------------|---------|
| `pd.DataFrame()` | Creates DataFrame from dict or list | `pd.DataFrame({'A': [1, 2], 'B': [3, 4]})` |
| `df.info()` | Shows DataFrame information | `df.info()` |
| `df.describe()` | Generates statistical summary | `df.describe()` |
| `df.dtypes` | Shows data types of columns | `df.dtypes` |
| `df.shape` | Returns dimensions of DataFrame | `df.shape` |
| `df.columns` | Lists column names | `df.columns` |
| `df.index` | Lists index labels | `df.index` |
| `df.head()` | Shows first n rows | `df.head(5)` |
| `df.tail()` | Shows last n rows | `df.tail(5)` |
| `df.sample()` | Returns random sample of rows | `df.sample(n=3)` |

## Data Access and Selection

| Function | Description | Example |
|----------|-------------|---------|
| `df.loc[]` | Label-based access | `df.loc['row1', 'col1']` |
| `df.iloc[]` | Integer-based access | `df.iloc[0, 0]` |
| `df['column']` | Selects single column | `df['name']` |
| `df[['col1', 'col2']]` | Selects multiple columns | `df[['name', 'age']]` |
| `df.isin()` | Checks for value presence | `df['col'].isin([1, 2, 3])` |
| `df.filter()` | Filters columns by name | `df.filter(like='name')` |

## Data Modification

| Function | Description | Example |
|----------|-------------|---------|
| `df.drop()` | Removes rows or columns | `df.drop('column', axis=1, inplace=True)` |
| `df.dropna()` | Removes missing values | `df.dropna(subset=['col'])` |
| `df.fillna()` | Fills missing values | `df.fillna(0)` |
| `df.replace()` | Replaces values | `df.replace(old_val, new_val)` |
| `df.rename()` | Renames columns or index | `df.rename(columns={'old': 'new'})` |
| `df.set_index()` | Sets index column | `df.set_index('column')` |
| `df.reset_index()` | Resets index | `df.reset_index()` |

## Sorting and Ranking

| Function | Description | Example |
|----------|-------------|---------|
| `df.sort_values()` | Sorts by values | `df.sort_values('column', ascending=False)` |
| `df.sort_index()` | Sorts by index | `df.sort_index()` |
| `df.rank()` | Creates ranking | `df.rank(method='average')` |
| `df.nlargest()` | Gets n largest values | `df.nlargest(3, 'column')` |
| `df.nsmallest()` | Gets n smallest values | `df.nsmallest(3, 'column')` |

## Statistical Operations

| Function | Description | Example |
|----------|-------------|---------|
| `df.mean()` | Calculates mean | `df.mean(numeric_only=True)` |
| `df.median()` | Calculates median | `df.median()` |
| `df.mode()` | Finds mode | `df.mode()` |
| `df.std()` | Calculates standard deviation | `df.std()` |
| `df.var()` | Calculates variance | `df.var()` |
| `df.sum()` | Calculates sum | `df.sum()` |
| `df.count()` | Counts non-null values | `df.count()` |
| `df.min()` | Finds minimum | `df.min()` |
| `df.max()` | Finds maximum | `df.max()` |
| `df.cumsum()` | Calculates cumulative sum | `df.cumsum()` |
| `df.cumprod()` | Calculates cumulative product | `df.cumprod()` |

## Grouping and Aggregation

| Function | Description | Example |
|----------|-------------|---------|
| `df.groupby()` | Groups data | `df.groupby('column')` |
| `df.agg()` | Applies aggregation | `df.groupby('col').agg(['mean', 'sum'])` |
| `df.value_counts()` | Counts unique values | `df['column'].value_counts()` |
| `df.pivot_table()` | Creates pivot table | `df.pivot_table(values='val', index='idx')` |

## File Operations

| Function | Description | Example |
|----------|-------------|---------|
| `pd.read_csv()` | Reads CSV file | `pd.read_csv('file.csv', sep=',')` |
| `df.to_csv()` | Writes to CSV | `df.to_csv('file.csv', index=False)` |
| `pd.read_json()` | Reads JSON file | `pd.read_json('file.json')` |
| `df.to_json()` | Writes to JSON | `df.to_json('file.json')` |
| `pd.read_excel()` | Reads Excel file | `pd.read_excel('file.xlsx')` |
| `df.to_excel()` | Writes to Excel | `df.to_excel('file.xlsx')` |

## Advanced Operations

| Function | Description | Example |
|----------|-------------|---------|
| `df.apply()` | Applies function to data | `df.apply(lambda x: x.max())` |
| `df.corr()` | Calculates correlation | `df.corr(numeric_only=True)` |
| `df.merge()` | Merges DataFrames | `df1.merge(df2, on='key')` |
| `df.join()` | Joins DataFrames | `df1.join(df2)` |
| `df.concat()` | Concatenates DataFrames | `pd.concat([df1, df2])` |
| `df.melt()` | Reshapes wide to long | `df.melt(id_vars=['id'])` |
| `df.pivot()` | Reshapes long to wide | `df.pivot(index='idx', columns='cols')` |

## Data Cleaning

| Function | Description | Example |
|----------|-------------|---------|
| `df.duplicated()` | Finds duplicate rows | `df.duplicated()` |
| `df.drop_duplicates()` | Removes duplicates | `df.drop_duplicates()` |
| `df.astype()` | Converts data type | `df['col'].astype('int64')` |
| `df.copy()` | Creates DataFrame copy | `df.copy()` |
| `df.interpolate()` | Interpolates missing values | `df.interpolate()` |
| `df.isna()` | Checks for missing values | `df.isna()` |
| `df.notna()` | Checks for non-missing values | `df.notna()` |

# Numpy functions
# Complete NumPy Functions Reference

## Array Creation and Conversion Functions

| Function | Description | Example | Output/Result |
|----------|-------------|---------|---------------|
| `np.array()` | Create array | `np.array([1, 2, 3])` | `[1, 2, 3]` |
| `np.asarray()` | Convert to array | `np.asarray(list)` | Array from list |
| `np.arange()` | Range array | `np.arange(3)` | `[0, 1, 2]` |
| `np.linspace()` | Evenly spaced | `np.linspace(0, 1, 5)` | 5 evenly spaced numbers |
| `np.logspace()` | Log-spaced | `np.logspace(0, 2, 3)` | `[1, 10, 100]` |
| `np.zeros()` | Array of zeros | `np.zeros((2, 2))` | 2x2 array of zeros |
| `np.ones()` | Array of ones | `np.ones((2, 2))` | 2x2 array of ones |
| `np.empty()` | Uninitialized | `np.empty((2, 2))` | 2x2 uninitialized array |
| `np.full()` | Constant array | `np.full((2, 2), 7)` | 2x2 array of 7s |
| `np.eye()` | Identity matrix | `np.eye(3)` | 3x3 identity matrix |
| `np.identity()` | Identity matrix | `np.identity(3)` | 3x3 identity matrix |
| `np.zeros_like()` | Zeros like input | `np.zeros_like(array)` | Array of zeros |
| `np.ones_like()` | Ones like input | `np.ones_like(array)` | Array of ones |
| `np.full_like()` | Constant like input | `np.full_like(array, 7)` | Array of 7s |
| `np.fromfunction()` | From function | `np.fromfunction(lambda i,j: i+j, (2,2))` | Function-based array |
| `np.fromiter()` | From iterator | `np.fromiter(iter, dtype)` | Array from iterator |
| `np.frombuffer()` | From buffer | `np.frombuffer(buffer)` | Array from buffer |
| `np.meshgrid()` | Coordinate matrices | `np.meshgrid([1,2], [3,4])` | Coordinate arrays |
| `np.mgrid()` | Dense meshgrid | `np.mgrid[0:2, 0:2]` | Dense coordinate arrays |
| `np.ogrid()` | Open meshgrid | `np.ogrid[0:2, 0:2]` | Open coordinate arrays |

## Array Manipulation Functions

| Function | Description | Example | Output/Result |
|----------|-------------|---------|---------------|
| `reshape()` | Change shape | `array.reshape(2, 3)` | Reshaped array |
| `resize()` | Change size | `array.resize((2, 3))` | Resized array |
| `ravel()` | Flatten to 1D | `array.ravel()` | 1D array |
| `flatten()` | Flatten copy | `array.flatten()` | 1D array copy |
| `transpose()` | Transpose axes | `array.transpose()` | Transposed array |
| `swapaxes()` | Swap axes | `array.swapaxes(0, 1)` | Axes swapped |
| `moveaxis()` | Move axes | `np.moveaxis(array, 0, 1)` | Axes moved |
| `rollaxis()` | Roll axis | `np.rollaxis(array, 2)` | Axis rolled |
| `expand_dims()` | Add dimension | `np.expand_dims(array, 0)` | Expanded array |
| `squeeze()` | Remove single-dims | `np.squeeze(array)` | Squeezed array |
| `broadcast_to()` | Broadcast array | `np.broadcast_to(array, shape)` | Broadcasted array |
| `broadcast_arrays()` | Multiple broadcast | `np.broadcast_arrays(a1, a2)` | Multiple arrays |
| `atleast_1d()` | Ensure 1D | `np.atleast_1d(array)` | At least 1D array |
| `atleast_2d()` | Ensure 2D | `np.atleast_2d(array)` | At least 2D array |
| `atleast_3d()` | Ensure 3D | `np.atleast_3d(array)` | At least 3D array |
| `flip()` | Reverse elements | `np.flip(array)` | Reversed array |
| `fliplr()` | Flip left/right | `np.fliplr(array)` | Left-right flipped |
| `flipud()` | Flip up/down | `np.flipud(array)` | Up-down flipped |
| `rot90()` | Rotate 90° | `np.rot90(array)` | Rotated array |
| `roll()` | Roll elements | `np.roll(array, 2)` | Rolled array |

## Joining and Splitting Functions

| Function | Description | Example | Output/Result |
|----------|-------------|---------|---------------|
| `concatenate()` | Join arrays | `np.concatenate((a1, a2))` | Combined array |
| `vstack()` | Stack vertically | `np.vstack((a1, a2))` | Vertical stack |
| `hstack()` | Stack horizontally | `np.hstack((a1, a2))` | Horizontal stack |
| `dstack()` | Stack depth-wise | `np.dstack((a1, a2))` | Depth stack |
| `stack()` | Stack along axis | `np.stack((a1, a2))` | Stacked array |
| `column_stack()` | Stack as columns | `np.column_stack((a1, a2))` | Column stack |
| `row_stack()` | Stack as rows | `np.row_stack((a1, a2))` | Row stack |
| `split()` | Split array | `np.split(array, 3)` | Split parts |
| `vsplit()` | Split vertically | `np.vsplit(array, 2)` | Vertical splits |
| `hsplit()` | Split horizontally | `np.hsplit(array, 2)` | Horizontal splits |
| `dsplit()` | Split depth-wise | `np.dsplit(array, 2)` | Depth splits |
| `array_split()` | Split unevenly | `np.array_split(array, 3)` | Uneven splits |

## Mathematical Functions

| Function | Description | Example | Output/Result |
|----------|-------------|---------|---------------|
| `add()` | Addition | `np.add(x1, x2)` | Sum |
| `subtract()` | Subtraction | `np.subtract(x1, x2)` | Difference |
| `multiply()` | Multiplication | `np.multiply(x1, x2)` | Product |
| `divide()` | Division | `np.divide(x1, x2)` | Quotient |
| `power()` | Power | `np.power(x1, x2)` | Power |
| `sqrt()` | Square root | `np.sqrt(x)` | Square root |
| `square()` | Square | `np.square(x)` | Square |
| `absolute()` | Absolute value | `np.absolute(x)` | Absolute value |
| `fabs()` | Absolute (float) | `np.fabs(x)` | Absolute (float) |
| `sign()` | Sign indicator | `np.sign(x)` | Signs |
| `ceil()` | Ceiling | `np.ceil(x)` | Ceiling |
| `floor()` | Floor | `np.floor(x)` | Floor |
| `rint()` | Round to integer | `np.rint(x)` | Rounded |
| `exp()` | Exponential | `np.exp(x)` | e^x |
| `exp2()` | 2^x | `np.exp2(x)` | 2^x |
| `log()` | Natural log | `np.log(x)` | ln(x) |
| `log2()` | Base-2 log | `np.log2(x)` | log2(x) |
| `log10()` | Base-10 log | `np.log10(x)` | log10(x) |
| `expm1()` | exp(x)-1 | `np.expm1(x)` | e^x - 1 |
| `log1p()` | log(1+x) | `np.log1p(x)` | ln(1+x) |

## Trigonometric Functions

| Function | Description | Example | Output/Result |
|----------|-------------|---------|---------------|
| `sin()` | Sine | `np.sin(x)` | sin(x) |
| `cos()` | Cosine | `np.cos(x)` | cos(x) |
| `tan()` | Tangent | `np.tan(x)` | tan(x) |
| `arcsin()` | Inverse sine | `np.arcsin(x)` | arcsin(x) |
| `arccos()` | Inverse cosine | `np.arccos(x)` | arccos(x) |
| `arctan()` | Inverse tangent | `np.arctan(x)` | arctan(x) |
| `hypot()` | Hypotenuse | `np.hypot(x, y)` | sqrt(x^2 + y^2) |
| `degrees()` | Radians to degrees | `np.degrees(x)` | Degrees |
| `radians()` | Degrees to radians | `np.radians(x)` | Radians |
| `deg2rad()` | Degrees to radians | `np.deg2rad(x)` | Radians |
| `rad2deg()` | Radians to degrees | `np.rad2deg(x)` | Degrees |

## Statistical Functions

| Function | Description | Example | Output/Result |
|----------|-------------|---------|---------------|
| `amin()` | Array minimum | `np.amin(array)` | Minimum |
| `amax()` | Array maximum | `np.amax(array)` | Maximum |
| `nanmin()` | Min ignoring NaN | `np.nanmin(array)` | Minimum |
| `nanmax()` | Max ignoring NaN | `np.nanmax(array)` | Maximum |
| `ptp()` | Peak to peak | `np.ptp(array)` | Range |
| `percentile()` | Percentile | `np.percentile(array, 50)` | 50th percentile |
| `nanpercentile()` | Percentile no NaN | `np.nanpercentile(array, 50)` | 50th percentile |
| `quantile()` | Quantile | `np.quantile(array, 0.5)` | 0.5 quantile |
| `nanquantile()` | Quantile no NaN | `np.nanquantile(array, 0.5)` | 0.5 quantile |
| `median()` | Median | `np.median(array)` | Median |
| `nanmedian()` | Median no NaN | `np.nanmedian(array)` | Median |
| `mean()` | Mean | `np.mean(array)` | Mean |
| `nanmean()` | Mean no NaN | `np.nanmean(array)` | Mean |
| `std()` | Standard deviation | `np.std(array)` | Std dev |
| `nanstd()` | Std dev no NaN | `np.nanstd(array)` | Std dev |
| `var()` | Variance | `np.var(array)` | Variance |
| `nanvar()` | Variance no NaN | `np.nanvar(array)` | Variance |
| `corrcoef()` | Correlation coef | `np.corrcoef(x, y)` | Correlation |
| `correlate()` | Cross-correlation | `np.correlate(x, y)` | Cross-correlation |
| `cov()` | Covariance | `np.cov(x, y)` | Covariance |

## Linear Algebra Functions

| Function | Description | Example | Output/Result |
|----------|-------------|---------|---------------|
| `dot()` | Dot product | `np.dot(a, b)` | Dot product |
| `vdot()` | Vector dot product | `np.vdot(a, b)` | Scalar product |
| `inner()` | Inner product | `np.inner(a, b)` | Inner product |
| `outer()` | Outer product | `np.outer(a, b)` | Outer product |
| `matmul()` | Matrix product | `np.matmul(a, b)` | Matrix product |
| `determinant()` | Matrix determinant | `np.linalg.det(a)` | Determinant |
| `solve()` | Solve linear eqn | `np.linalg.solve(a, b)` | Solution |
| `inv()` | Matrix inverse | `np.linalg.inv(a)` | Inverse |
| `pinv()` | Pseudo-inverse | `np.linalg.pinv(a)` | Pseudo-inverse |
| `qr()` | QR decomposition | `np.linalg.qr(a)` | Q, R matrices |
| `svd()` | Singular values | `np.linalg.svd(a)` | U, S, V matrices |
| `eig()` | Eigenvalues | `np.linalg.eig(a)` | Eigenval/vec |
| `eigvals()` | Eigenvalues only | `np.linalg.eigvals(a)` | Eigenvalues |
| `norm()` | Matrix/vector norm | `np.linalg.norm(a)` | Norm |
| `matrix_rank()` | Matrix rank | `np.linalg.matrix_rank(a)` | Rank |
| `trace()` | Matrix trace | `np.trace(a)` | Trace |
| `kron()` | Kronecker product | `np.kron(a, b)` | Kronecker product |
| `matrix_power()` | Matrix power | `np.linalg.matrix_power(a, n)` | Matrix^n |

## Logical Functions

| Function | Description | Example | Output/Result |
|----------|-------------|---------|---------------|
| `all()` | Test if all true | `np.all(array)` | Boolean |
| `any()` | Test if any true | `np.any(array)` | Boolean |
| `isfinite()` | Test finite | `np.isfinite(array)` | Boolean array |
| `isinf()` | Test infinite | `np.isinf(array)` | Boolean array |
| `isnan()` | Test NaN | `np.isnan(array)` | Boolean array |
| `isneginf()` | Test negative inf | `np.isneginf(array)` | Boolean

# Complete Pandas Series Reference

## Series Creation
```python
# Basic creation
s = pd.Series([1, 2, 3])
s = pd.Series(data=[1, 2, 3], index=['a', 'b', 'c'])
s = pd.Series({'a': 1, 'b': 2, 'c': 3})
s = pd.Series(5, index=['a', 'b', 'c'])  # Constant value
```

## Basic Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `values` | Underlying data as array | `s.values` |
| `index` | Index object | `s.index` |
| `dtype` | Data type | `s.dtype` |
| `shape` | Shape of Series | `s.shape` |
| `size` | Number of elements | `s.size` |
| `ndim` | Number of dimensions | `s.ndim` |
| `nbytes` | Memory usage in bytes | `s.nbytes` |
| `name` | Series name | `s.name` |
| `empty` | Whether Series is empty | `s.empty` |
| `hasnans` | Whether has NaN values | `s.hasnans` |
| `axes` | Return list of axes | `s.axes` |
| `T` | Transpose | `s.T` |

## Accessing Data

| Method | Description | Example |
|--------|-------------|---------|
| `at[]` | Access single value by label | `s.at['label']` |
| `iat[]` | Access single value by position | `s.iat[0]` |
| `loc[]` | Access values by label | `s.loc['a':'c']` |
| `iloc[]` | Access values by position | `s.iloc[0:3]` |
| `items()` | Iterator over (index, value) pairs | `for idx, val in s.items()` |
| `keys()` | Get index | `s.keys()` |
| `get()` | Get value with default for missing | `s.get('key', default=0)` |
| `xs()` | Access single value by label | `s.xs('label')` |

## Information Methods

| Method | Description | Example |
|--------|-------------|---------|
| `describe()` | Summary statistics | `s.describe()` |
| `info()` | Series information | `s.info()` |
| `count()` | Count non-NA/null values | `s.count()` |
| `value_counts()` | Count unique values | `s.value_counts()` |
| `nunique()` | Count number of unique values | `s.nunique()` |
| `unique()` | Get unique values | `s.unique()` |
| `memory_usage()` | Memory usage in bytes | `s.memory_usage()` |
| `is_unique` | Check if values are unique | `s.is_unique` |
| `is_monotonic` | Check if values are monotonic | `s.is_monotonic` |
| `is_monotonic_increasing` | Check if increasing | `s.is_monotonic_increasing` |
| `is_monotonic_decreasing` | Check if decreasing | `s.is_monotonic_decreasing` |

## Mathematical Methods

| Method | Description | Example |
|--------|-------------|---------|
| `abs()` | Absolute values | `s.abs()` |
| `add()` | Addition | `s.add(other)` |
| `sub()` | Subtraction | `s.sub(other)` |
| `mul()` | Multiplication | `s.mul(other)` |
| `div()` | Division | `s.div(other)` |
| `truediv()` | True division | `s.truediv(other)` |
| `floordiv()` | Floor division | `s.floordiv(other)` |
| `mod()` | Modulo | `s.mod(other)` |
| `pow()` | Power | `s.pow(other)` |
| `radd()` | Reverse addition | `s.radd(other)` |
| `rsub()` | Reverse subtraction | `s.rsub(other)` |
| `rmul()` | Reverse multiplication | `s.rmul(other)` |
| `rdiv()` | Reverse division | `s.rdiv(other)` |
| `round()` | Round values | `s.round(decimals=2)` |
| `clip()` | Trim values | `s.clip(lower=0, upper=1)` |

## Statistical Methods

| Method | Description | Example |
|--------|-------------|---------|
| `mean()` | Mean | `s.mean()` |
| `median()` | Median | `s.median()` |
| `mode()` | Mode | `s.mode()` |
| `var()` | Variance | `s.var()` |
| `std()` | Standard deviation | `s.std()` |
| `sem()` | Standard error of mean | `s.sem()` |
| `skew()` | Skewness | `s.skew()` |
| `kurt()` | Kurtosis | `s.kurt()` |
| `quantile()` | Quantile | `s.quantile(0.5)` |
| `min()` | Minimum | `s.min()` |
| `max()` | Maximum | `s.max()` |
| `sum()` | Sum | `s.sum()` |
| `prod()` | Product | `s.prod()` |
| `compound()` | Compound percentage | `s.compound()` |
| `cumsum()` | Cumulative sum | `s.cumsum()` |
| `cumprod()` | Cumulative product | `s.cumprod()` |
| `cummax()` | Cumulative maximum | `s.cummax()` |
| `cummin()` | Cumulative minimum | `s.cummin()` |
| `diff()` | Difference | `s.diff()` |
| `pct_change()` | Percentage change | `s.pct_change()` |
| `rank()` | Rank | `s.rank()` |
| `nlargest()` | n largest values | `s.nlargest(n=3)` |
| `nsmallest()` | n smallest values | `s.nsmallest(n=3)` |

## Missing Data Methods

| Method | Description | Example |
|--------|-------------|---------|
| `isna()` | Detect missing values | `s.isna()` |
| `notna()` | Detect non-missing values | `s.notna()` |
| `isnull()` | Alias for isna | `s.isnull()` |
| `notnull()` | Alias for notna | `s.notnull()` |
| `dropna()` | Drop missing values | `s.dropna()` |
| `fillna()` | Fill missing values | `s.fillna(0)` |
| `interpolate()` | Interpolate values | `s.interpolate()` |
| `bfill()` | Backward fill | `s.bfill()` |
| `ffill()` | Forward fill | `s.ffill()` |
| `pad()` | Alias for ffill | `s.pad()` |
| `backfill()` | Alias for bfill | `s.backfill()` |

## Transformation Methods

| Method | Description | Example |
|--------|-------------|---------|
| `map()` | Map values using input | `s.map({'a': 1, 'b': 2})` |
| `apply()` | Apply function | `s.apply(lambda x: x*2)` |
| `transform()` | Transform using function | `s.transform(lambda x: x+1)` |
| `replace()` | Replace values | `s.replace(1, 'one')` |
| `update()` | Modify Series in-place | `s1.update(s2)` |
| `mask()` | Replace values where condition | `s.mask(s > 0, 999)` |
| `where()` | Replace values where not condition | `s.where(s > 0, 999)` |
| `astype()` | Cast to dtype | `s.astype('int64')` |
| `convert_dtypes()` | Convert to best dtype | `s.convert_dtypes()` |
| `infer_objects()` | Infer dtype | `s.infer_objects()` |
| `reindex()` | Conform to new index | `s.reindex(['a', 'b', 'c'])` |
| `rename()` | Rename Series | `s.rename('new_name')` |
| `set_axis()` | Set index or columns | `s.set_axis(['a', 'b', 'c'])` |

## Grouping and Window Methods

| Method | Description | Example |
|--------|-------------|---------|
| `groupby()` | Group Series | `s.groupby(level=0)` |
| `rolling()` | Rolling window | `s.rolling(window=3).mean()` |
| `expanding()` | Expanding window | `s.expanding().mean()` |
| `ewm()` | Exponential weighted window | `s.ewm(span=3).mean()` |
| `shift()` | Shift index | `s.shift(periods=1)` |
| `tshift()` | Time shift | `s.tshift(freq='D')` |
| `first()` | First element of group | `s.groupby(level=0).first()` |
| `last()` | Last element of group | `s.groupby(level=0).last()` |
| `nth()` | nth element of group | `s.groupby(level=0).nth(2)` |
| `resample()` | Resample time-series data | `s.resample('D').mean()` |

## Combining Methods

| Method | Description | Example |
|--------|-------------|---------|
| `append()` | Append Series | `s1.append(s2)` |
| `combine()` | Combine Series | `s1.combine(s2, func=max)` |
| `combine_first()` | Update null elements | `s1.combine_first(s2)` |
| `align()` | Align Series | `s1.align(s2)` |
| `compare()` | Compare with other | `s1.compare(s2)` |
| `equals()` | Test equality | `s1.equals(s2)` |

## String Methods (str accessor)

| Method | Description | Example |
|--------|-------------|---------|
| `str.lower()` | Lowercase | `s.str.lower()` |
| `str.upper()` | Uppercase | `s.str.upper()` |
| `str.len()` | String length | `s.str.len()` |
| `str.strip()` | Strip whitespace | `s.str.strip()` |
| `str.split()` | Split string | `s.str.split(',')` |
| `str.replace()` | Replace substring | `s.str.replace('old', 'new')` |
| `str.contains()` | Test if contains | `s.str.contains('pattern')` |
| `str.startswith()` | Test if starts with | `s.str.startswith('prefix')` |
| `str.endswith()` | Test if ends with | `s.str.endswith('suffix')` |
| `str.cat()` | Concatenate strings | `s.str.cat(sep=',')` |
| `str.extract()` | Extract using regex | `s.str.extract('(\d+)')` |
| `str.find()` | Find substring | `s.str.find('pattern')` |
| `str.pad()` | Pad strings | `s.str.pad(width=10)` |
| `str.slice()` | Slice strings | `s.str.slice(0, 3)` |

## Datetime Methods (dt accessor)

| Method | Description | Example |
|--------|-------------|---------|
| `dt.year` | Extract year | `s.dt.year` |
| `dt.month` | Extract month | `s.dt.month` |
| `dt.day` | Extract day | `s.dt.day` |
| `dt.hour` | Extract hour | `s.dt.hour` |
| `dt.minute` | Extract minute | `s.dt.minute` |
| `dt.second` | Extract second | `s.dt.second` |
| `dt.weekday` | Extract weekday | `s.dt.weekday` |
| `dt.dayofyear` | Day of year | `s.dt.dayofyear` |
| `dt.quarter` | Extract quarter | `s.dt.quarter` |
| `dt.strftime()` | Format datetime | `s.dt.strftime('%Y-%m-%d')` |
| `dt.tz_localize()` | Localize timezone | `s.dt.tz_localize('UTC')` |
| `dt.tz_convert()` | Convert timezone | `s.dt.tz_convert('US/Pacific')` |

## Input/Output Methods

| Method | Description | Example |
|--------|-------------|---------|
| `to_numpy()` | Convert to numpy array | `s.to_numpy()` |
| `to_list()` | Convert to list | `s.to_list()` |
| `to_dict()` | Convert to dict | `s.to_dict()` |
| `to_frame()` | Convert to DataFrame | `s.to_frame()` |
| `to_string()` | Convert to string | `s.to_string()` |
| `to_json()` | Convert to JSON | `s.to_json()` |
| `to_csv()` | Write to CSV | `s.to_csv('file.csv')` |
| `to_excel()` | Write to Excel | `s.to_excel('file.xlsx')` |
| `to_pickle()` | Write to pickle | `s.to_pickle('file.pkl')` |

## Utility Methods

| Method | Description | Example |
|--------|-------------|---------|
| `copy()` | Create copy | `s.copy()` |
| `pipe()` | Apply func chainable | `s.pipe(func)` |
| `drop()` | Drop labels | `s.drop(['a', 'b'])` |
| `droplevel()` | Drop index levels | `s.droplevel(0)` |
| `reset_index()` | Reset index | `s.reset_index()` |
| `set_flags()` | Set flags | `s.set_flags(allows_duplicate_labels=False)` |
| `between()` | Check if between | `s.between(left=0, right=1)` |
| `sort_values()` | Sort by values | `s.sort_values()` |
| `sort_index()` | Sort by index | `s.sort_index()` |
| `reorder_levels()` | Reorder levels | `s.reorder_levels([1, 0])` |
| `swaplevel()` | Swap levels | `s.swaplevel(0, 1)` |

# More Pandas

# Pandas Functions Reference Table

## Data Creation Functions

| Function | Description | Example |
|----------|-------------|---------|
| `pd.Series()` | Creates a 1D labeled array | `pd.Series([1, 2, 3], index=['a', 'b', 'c'])` |
| `pd.DataFrame()` | Creates a 2D labeled data structure | `pd.DataFrame({'A': [1, 2], 'B': [3, 4]})` |
| `pd.date_range()` | Creates date series | `pd.date_range('20230101', periods=6)` |
| `pd.Index()` | Creates an index object | `pd.Index([2, 3, 5, 23, 26])` |

## Data Access Functions

| Function | Description | Example |
|----------|-------------|---------|
| `df.loc[]` | Label-based access | `df.loc['row_label', 'column_label']` |
| `df.iloc[]` | Integer-based access | `df.iloc[0, 0]` |
| `df['column']` | Column access | `df['column_name']` |
| `df.at[]` | Fast label-based scalar accessor | `df.at['row_label', 'column_label']` |
| `df.iat[]` | Fast integer-based scalar accessor | `df.iat[0, 0]` |
| `df.head()` | First n rows | `df.head(5)` |
| `df.tail()` | Last n rows | `df.tail(5)` |

## Lambda Operations

| Function | Description | Example |
|----------|-------------|---------|
| `apply(lambda)` by row | Apply function to each row | `df.apply(lambda row: row['A'] + row['B'], axis=1)` |
| `apply(lambda)` by column | Apply function to each column | `df.apply(lambda col: col.max() - col.min(), axis=0)` |
| Row-wise calculation | Perform operation on each row | `df.apply(lambda x: x.sum(), axis=1)` |
| Column-wise calculation | Perform operation on each column | `df.apply(lambda x: x.mean(), axis=0)` |
| Conditional lambda | Apply condition to elements | `df['A'].apply(lambda x: 'High' if x > 5 else 'Low')` |
| Multiple column lambda | Operation using multiple columns | `df.apply(lambda x: x['A']/x['B'] if x['B']!=0 else 0, axis=1)` |
| Filter lambda | Filter groups using lambda | `df.groupby('A').filter(lambda x: x['B'].mean() > 30)` |

## Data Manipulation Functions

| Function | Description | Example |
|----------|-------------|---------|
| `sort_values()` | Sort by values | `df.sort_values('column', ascending=False)` |
| `sort_index()` | Sort by index | `df.sort_index(ascending=True)` |
| `rank()` | Compute numerical rank | `df.rank(method='average', axis=0)` |
| `groupby()` | Group data | `df.groupby('column').mean()` |
| `merge()` | Merge DataFrames | `pd.merge(df1, df2, on='key')` |
| `concat()` | Concatenate DataFrames | `pd.concat([df1, df2])` |
| `drop()` | Drop rows or columns | `df.drop('column', axis=1)` |
| `rename()` | Rename columns | `df.rename(columns={'old': 'new'})` |
| `reset_index()` | Reset index | `df.reset_index(drop=True)` |
| `set_index()` | Set index | `df.set_index('column')` |
| `swapaxes()` | Swap axes | `df.swapaxes(0,1)` |
| `T` | Transpose | `df.T` |

## Statistical Functions

| Function | Description | Example |
|----------|-------------|---------|
| `describe()` | Statistical summary | `df.describe()` |
| `mean()` | Calculate mean | `df.mean(numeric_only=True)` |
| `median()` | Calculate median | `df.median()` |
| `std()` | Calculate standard deviation | `df.std()` |
| `var()` | Calculate variance | `df.var()` |
| `mad()` | Mean absolute deviation | `df.mad()` |
| `corr()` | Calculate correlation | `df.corr()` |
| `count()` | Count non-null values | `df.count()` |
| `max()` | Maximum value | `df.max()` |
| `min()` | Minimum value | `df.min()` |
| `sum()` | Sum of values | `df.sum()` |
| `cumsum()` | Cumulative sum | `df.cumsum()` |
| `cumprod()` | Cumulative product | `df.cumprod()` |
| `argmax()` | Index of maximum value | `df.argmax()` |
| `argmin()` | Index of minimum value | `df.argmin()` |
| `idxmax()` | Returns index label of maximum value | `df.idxmax()` # By column\ndf.idxmax(axis=1) # By row\ndf.idxmax(skipna=False) # Include NaN` |
| `idxmin()` | Returns index label of minimum value | `df.idxmin()` # By column\ndf.idxmin(skipna=True) # Skip NaN\ndf['col'].idxmin() # For single column` |
| `argmax()` | Returns integer position of maximum value | `df.argmax() # Position by column\ndf.argmax(axis=1) # Position by row` |
| `argmin()` | Returns integer position of minimum value | `df.argmin() # Position by column\ns.argmin() # For series` |

## Data Cleaning Functions

### Missing Values
| Function | Description | Example |
|----------|-------------|---------|
| `isnull()` | Check for missing values | `df.isnull()` |
| `notnull()` | Check for non-missing values | `df.notnull()` |
| `dropna()` | Drop missing values | `df.dropna(subset=['col'])`\n`df.dropna(how='all')`\n`df.dropna(thresh=2)` |
| `fillna()` | Fill missing values | `df.fillna(0)`\n`df.fillna(method='ffill')`\n`df.fillna(method='bfill')` |
| `replace()` | Replace values | `df.replace(old_val, new_val)`\n`df.replace([1,2], np.nan)` |
| `pd.notnull()` | Check for non-null values | `pd.notnull(df)` |
| `interpolate()` | Interpolate values | `df.interpolate(method='linear')`\n`df.interpolate(method='pad')` |

### Duplicates
| Function | Description | Example |
|----------|-------------|---------|
| `duplicated()` | Return boolean Series denoting duplicate rows | `df.duplicated()`\n`df.duplicated(subset=['A', 'B'])`\n`df.duplicated(keep='last')` |
| `drop_duplicates()` | Remove duplicate rows | `df.drop_duplicates()`\n`df.drop_duplicates(subset=['A'])`\n`df.drop_duplicates(keep='last')` |

### Data Type Handling
| Function | Description | Example |
|----------|-------------|---------|
| `astype()` | Convert data type | `df['col'].astype('int64')`\n`df.astype({'col1': 'int32', 'col2': 'float64'})` |
| `convert_dtypes()` | Convert to best possible dtypes | `df.convert_dtypes()` |
| `infer_objects()` | Infer better data types | `df.infer_objects()` |

### Value Validation
| Function | Description | Example |
|----------|-------------|---------|
| `isna()` | Detect missing values | `df.isna().sum()`\n`df.isna().any()` |
| `notna()` | Detect non-missing values | `df.notna().all()` |
| `between()` | Check if values are between bounds | `df['col'].between(left=0, right=10)` |
| `clip()` | Trim values at bounds | `df.clip(lower=0, upper=100)` |
| `round()` | Round values | `df.round(2)` |

## File Operations

| Function | Description | Example |
|----------|-------------|---------|
| `read_csv()` | Read CSV file | `pd.read_csv('file.csv', sep=',', header=None, names=['A','B'])` |
| `to_csv()` | Write to CSV | `df.to_csv('file.csv', index=False, na_rep='NULL')` |
| `read_json()` | Read JSON file | `pd.read_json('file.json')` |
| `to_json()` | Write to JSON | `df.to_json('file.json')` |
| `read_excel()` | Read Excel file | `pd.read_excel('file.xlsx')` |
| `to_excel()` | Write to Excel | `df.to_excel('file.xlsx')` |
| `read_table()` | Read general delimited file | `pd.read_table('file.txt', sep='\t')` |

## Selection and Filtering Functions

| Function | Description | Example |
|----------|-------------|---------|
| `filter()` | Filter data | `df.filter(items=['A', 'B'])` |
| `isin()` | Check for values | `df['A'].isin([1, 2])` |
| `where()` | Replace values conditionally | `df.where(df > 0, 0)` |
| `query()` | Query DataFrame | `df.query('A > B')` |
| `sample()` | Random sample | `df.sample(n=5)` |
| `nlargest()` | Get n largest values | `df.nlargest(3, 'column')` |
| `nsmallest()` | Get n smallest values | `df.nsmallest(3, 'column')` |
| `mask()` | Replace values using mask | `df.mask(df < 0, 0)` |
| `between()` | Check if values are between | `df['A'].between(left=2, right=5)` |

## Aggregation and Grouping Functions

| Function | Description | Example |
|----------|-------------|---------|
| `agg()` | Aggregate using functions | `df.groupby('A').agg({'B': 'mean'})` |
| `transform()` | Transform groups | `df.groupby('A').transform('sum')` |
| `rolling()` | Rolling window calculations | `df.rolling(window=3).mean()` |
| `expanding()` | Expanding window calculations | `df.expanding().mean()` |
| `pivot_table()` | Create pivot table | `df.pivot_table(values='D', index='A')` |
| `melt()` | Unpivot table | `df.melt(id_vars=['A'])` |
| `value_counts()` | Count unique values | `df['column'].value_counts()` |
| `aggregate()` | Multiple function aggregation | `df.groupby('A').aggregate(['min', 'max', 'mean'])` |
| `resample()` | Resample time-series data | `df.resample('D').mean()` |

## DataFrame Information Functions

| Function | Description | Example |
|----------|-------------|---------|
| `info()` | DataFrame information | `df.info()` |
| `dtypes` | Get column data types | `df.dtypes` |
| `shape` | Get dimensions | `df.shape` |
| `size` | Get size | `df.size` |
| `copy()` | Create copy | `df.copy()` |
| `memory_usage()` | Memory usage | `df.memory_usage(deep=True)` |
| `nunique()` | Count unique values | `df.nunique()` |
| `items()` | Iterator over (column, Series) pairs | `for col, series in df.items()` |
| `equals()` | Test equality with another object | `df1.equals(df2)` |
| `empty` | True if DataFrame is empty | `df.empty` |
| `ndim` | Number of dimensions | `df.ndim` |
| `axes` | List of axes labels | `df.axes` |
| `select_dtypes()` | Select columns by data type | `df.select_dtypes(include=['number'])` |
| `get_numeric_data()` | Get numeric columns | `df.get_numeric_data()` |
| `columns` | Get column labels | `df.columns` |
| `index` | Get index labels | `df.index` |
| `values` | Get numpy array of values | `df.values` |
| `to_numpy()` | Convert to numpy array | `df.to_numpy()` |

## String Operations (str accessor)

| Function | Description | Example |
|----------|-------------|---------|
| `str.lower()` | Convert to lowercase | `df['col'].str.lower()` |
| `str.upper()` | Convert to uppercase | `df['col'].str.upper()` |
| `str.len()` | Get string length | `df['col'].str.len()` |
| `str.strip()` | Remove whitespace | `df['col'].str.strip()` |
| `str.split()` | Split string | `df['col'].str.split(',')` |
| `str.replace()` | Replace pattern | `df['col'].str.replace('old', 'new')` |
| `str.contains()` | Check if string contains pattern | `df['col'].str.contains('pattern')` |
| `str.startswith()` | Check string start | `df['col'].str.startswith('prefix')` |
| `str.endswith()` | Check string end | `df['col'].str.endswith('suffix')` |
| `str.cat()` | Concatenate strings | `df['col'].str.cat(sep=', ')` |
| `str.extract()` | Extract pattern matches | `df['col'].str.extract('(\d+)')` |
| `str.get()` | Get element at position | `df['col'].str.get(0)` |

## Time Series Operations (dt accessor)

| Function | Description | Example |
|----------|-------------|---------|
| `dt.year` | Get year | `df['date'].dt.year` |
| `dt.month` | Get month | `df['date'].dt.month` |
| `dt.day` | Get day | `df['date'].dt.day` |
| `dt.hour` | Get hour | `df['date'].dt.hour` |
| `dt.minute` | Get minute | `df['date'].dt.minute` |
| `dt.second` | Get second | `df['date'].dt.second` |
| `dt.weekday` | Get day of week | `df['date'].dt.weekday` |
| `dt.dayofyear` | Get day of year | `df['date'].dt.dayofyear` |
| `dt.quarter` | Get quarter | `df['date'].dt.quarter` |
| `dt.strftime()` | Format datetime | `df['date'].dt.strftime('%Y-%m-%d')` |

## Data Reshaping

| Function | Description | Example |
|----------|-------------|---------|
| `pivot()` | Reshape data | `df.pivot(index='A', columns='B', values='C')` |
| `pivot_table()` | Create pivot table | `df.pivot_table(values='D', index=['A', 'B'], aggfunc='sum')` |
| `melt()` | Unpivot DataFrame | `df.melt(id_vars=['A'], value_vars=['B'])` |
| `stack()` | Stack the prescribed level(s) | `df.stack()` |
| `unstack()` | Unstack the prescribed level(s) | `df.unstack()` |
| `wide_to_long()` | Wide format to long format | `pd.wide_to_long(df, stubnames='val', i=['id'], j='time')` |

## Window Operations

| Function | Description | Example |
|----------|-------------|---------|
| `rolling()` | Rolling window calculations | `df.rolling(window=3).mean()` |
| `expanding()` | Expanding window calculations | `df.expanding().sum()` |
| `ewm()` | Exponential weighted calculations | `df.ewm(span=3).mean()` |
| `shift()` | Shift index | `df.shift(periods=1)` |
| `diff()` | Compute difference | `df.diff()` |
| `pct_change()` | Compute percentage change | `df.pct_change()` |

## Set Operations

| Function | Description | Example |
|----------|-------------|---------|
| `append()` | Append rows | `df1.append(df2)` |
| `combine_first()` | Update null elements | `df1.combine_first(df2)` |
| `update()` | Modify in place | `df1.update(df2)` |
| `assign()` | Assign new columns | `df.assign(D=lambda x: x['A'] * 10)` |
| `merge_ordered()` | Merge with optional filling | `pd.merge_ordered(df1, df2)` |
| `merge_asof()` | Merge on nearest key | `pd.merge_asof(df1, df2, on='date')` |
| `eval()` | Evaluate string expression | `df.eval('A + B')` |
| `query()` | Query DataFrame | `df.query('A > B')` |

## Computation Methods

| Function | Description | Example |
|----------|-------------|---------|
| `abs()` | Absolute values | `df.abs()` |
| `all()` | Check if all elements are True | `df.all()` |
| `any()` | Check if any elements are True | `df.any()` |
| `clip()` | Trim values | `df.clip(lower=0, upper=1)` |
| `prod()` | Product of values | `df.prod()` |
| `compound()` | Compound percentage | `df.compound()` |
| `cummax()` | Cumulative maximum | `df.cummax()` |
| `cummin()` | Cumulative minimum | `df.cummin()` |
| `cumprod()` | Cumulative product | `df.cumprod()` |
| `cumsum()` | Cumulative sum | `df.cumsum()` |
| `dot()` | Matrix multiplication | `df.dot(other_df)` |
| `mode()` | Mode of values | `df.mode()` |
| `quantile()` | Compute quantiles | `df.quantile([0.25, 0.75])` |
| `rank()` | Compute numerical rank | `df.rank()` |
| `round()` | Round values | `df.round(2)` |
| `sem()` | Standard error of mean | `df.sem()` |
| `skew()` | Sample skewness | `df.skew()` |
| `kurt()` | Kurtosis | `df.kurt()` |

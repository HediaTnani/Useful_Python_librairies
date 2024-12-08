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

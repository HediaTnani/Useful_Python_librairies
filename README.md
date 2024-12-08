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

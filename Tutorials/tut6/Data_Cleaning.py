from pandas import read_excel

dataset = read_excel('synthetic-data.xlsx')
print(dataset)
print("Describe:")
print(dataset.describe())
print("Info:")
print(dataset.info())



missing_locs = dataset.isnull().to_numpy().nonzero()
row_indices = missing_locs[0]
column_indices = missing_locs[1]
for row_index, column_index in zip(row_indices, column_indices):
    print(f"Missing value at Row {row_index}, Column '{dataset.columns[column_index]}'")

duplicates = dataset.duplicated()
dup_locs = duplicates.to_numpy().nonzero()[0]
if dup_locs.size > 0:
    print(f"Duplicate value found at Rows {dup_locs}")

import pandas as pd
marks_dict={"jhon":78,"michael":85,"susan":92,"linda":88}
marks=pd.Series(marks_dict)
print(marks)
print(marks.values)
print(marks["jhon"])
print(marks[0:2])
print(marks[marks>80])
#indexing
print(marks.index)

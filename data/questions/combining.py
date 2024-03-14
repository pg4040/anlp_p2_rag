import pandas as pd
import numpy as np

np.random.seed(370)

df1 = pd.read_csv('../../History_data/qa_pairs/history_qa.csv')
df2 = pd.read_csv('../../faculty_data_subset/manual_questions.csv')
df2.rename(columns={' answer': 'answer', ' document': 'filepath'}, inplace=True)
df3 = pd.read_csv('../../Events_Data/qa_pairs/Commencement_QnA.csv')
df3 = df3.drop(columns=['Category'])
df3.rename(str.lower, axis='columns', inplace=True)
df4 = pd.read_csv('../../Events_Data/qa_pairs/QnA_data_events.csv')
df4 = df4.drop(columns=['Category'])
df4.rename(columns={'Question': 'question', 'Ref_Answer': 'answer', 'DocID': 'filepath'}, inplace=True)
random_rows_df4 = np.random.choice(df4.index, size=200, replace=False)
df4 = df4.loc[random_rows_df4]

df5 = pd.read_csv('../../Courses_data/qa_pairs/QnA_data_courses.csv')
df5 = df5.drop(columns=['Category'])
df5.rename(columns={'Question': 'question', 'Ref_Answer': 'answer', 'DocID': 'filepath'}, inplace=True)
random_rows_df5 = np.random.choice(df5.index, size=300, replace=False)
df5 = df5.loc[random_rows_df5]

df6 = pd.read_csv('../../Courses_data/qa_pairs/QnA_Data_Calendar_Courses.csv')
df6 = df6.drop(columns=['Category'])
df6.rename(columns={'Question': 'question', 'Answer': 'answer', 'DocID': 'filepath'}, inplace=True)
df7 = pd.read_csv('../../Academics_data/qa_pairs/Academics_Handbook_QnA.csv')
df7 = df7.drop(columns=['Category'])
df7.rename(str.lower, axis='columns', inplace=True)
df8 = pd.read_csv('../../Academics_data/qa_pairs/Academics_QnA.csv')
df8 = df8.drop(columns=['Category'])
df8.rename(str.lower, axis='columns', inplace=True)

# Concatenate all the dataframes vertically
combined_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])

# Write the combined dataframe to a .csv file
combined_df.to_csv('combined_data.csv', index=False)

# Write the first column to a .txt file (questions.txt)
combined_df.iloc[:, 0].to_csv('questions.txt', index=False, header=False)

# Write the second column to a .txt file (reference_answers.txt)
combined_df.iloc[:, 1].to_csv('reference_answers.txt', index=False, header=False)

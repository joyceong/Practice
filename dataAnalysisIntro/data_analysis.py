# -*- coding:utf-8 -*-
import unicodecsv
import pandas as pd


# ----- practise 1 -----
# 读取csv数据

enrollments = pd.read_csv('enrollments.csv')
daily_engagement = pd.read_csv('daily-engagement.csv')
project_submissions = pd.read_csv('project-submissions.csv')
print 'Practise 1: 读取csv数据成功！' 


# ----- practise 2 -----
# 找到3个csv表的总行数，以及不重复学员（account_key）的数量

enrollment_num_rows = len(enrollments['account_key'])
enrollment_num_unique_students = len(enrollments['account_key'].unique())

engagement_num_rows = len(daily_engagement['acct'])
engagement_num_unique_students = len(daily_engagement['acct'].unique())
submission_num_rows = len(project_submissions['account_key']) 
submission_num_unique_students = len(project_submissions['account_key'].unique())
print 'Practise 2:'
print '\tenrollments.csv总行数:', enrollment_num_rows, '不重复学员数:', enrollment_num_unique_students
print '\tdaily-engagement.csv总行数:', engagement_num_rows, '不重复学员数:', engagement_num_unique_students
print '\tproject-submissions.csv总行数:', submission_num_rows, '不重复学员数:', submission_num_unique_students


# ----- practise 3 -----
# 1. 重命名daily_engagement中的acct为account_key
# 2. 输出daily_engagement[0]['account_key']

daily_engagement.rename(columns={'acct':'account_key'}, inplace = True)
print 'Practise 3: \n\t1.已重命名\'acct\'为\'account_key\'\n\t2.daily_engagement[0][\'accout_key\']:', daily_engagement['account_key'][0]


# ----- practise 4 -----
# 找出enrollments中没有对应daily_engagement数据的account_key并统计个数

index2 = daily_engagement['account_key'].unique()
index1 = enrollments['account_key'].unique()
l = []
for i in index1:
    if i not in index2:
        l.append(i)
extra_enrollment = enrollments.iloc[l]
print 'Practise 4:'
print '\t异常数据数目:', len(extra_enrollment['account_key'].unique())


# ----- Practise 5 -----
# 找出cancel_date和join_date同一天但days_to_cancel!=0的数据-> null
print 'Practise 5: join_date == cancel_date 数目:',len(extra_enrollment[extra_enrollment.cancel_date==extra_enrollment.join_date]['account_key'].unique())
# is_udacity = enrollments.loc[enrollments['is_udacity']==True]
# print len(is_udacity['account_key'].unique())

# ----- practise 6 -----
# 创建paid_students，条件：days_to_cancel isnull或者>7
paid_students = enrollments[(enrollments.days_to_cancel.isnull())|(enrollments.days_to_cancel > 7)]
print 'Practise 6: paid_students人数:', len(paid_students)


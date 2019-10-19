from code_metrics import *
import os
import pandas as pd

input_dir = './dataset/'
df = pd.DataFrame()

i = 0
for file_name in os.listdir(input_dir):

	print(file_name)
	code = file_name
	if code=='.DS_Store':
		continue
	code = input_dir+code
    
    # LLC SHIT
	llc = line_length_calculator(code)
	for key in llc:
		df.loc[i,'llc_'+str(key)] = llc[key]
    
    # LWC SHIT
	lwc = line_word_calculator(code)
	for key in lwc:
		df.loc[i,'lwc_'+str(key)] = lwc[key]
        
    # HALSTEAD
	hal = halstead_metrics(code)
	df.loc[i,'h1'] = hal.h1
	df.loc[i,'h2'] = hal.h2
	df.loc[i,'N1'] = hal.N1
	df.loc[i,'N2'] = hal.N2
	df.loc[i,'vocabulary'] = hal.vocabulary
	df.loc[i,'length'] = hal.length
	df.loc[i,'calc_length'] = hal.calculated_length
	df.loc[i,'volume'] = hal.volume
	df.loc[i,'difficulty'] = hal.difficulty
	df.loc[i,'effort'] = hal.effort
	df.loc[i,'time'] = hal.time
	df.loc[i,'bugs'] = hal.bugs
    
    # RAW
	raw = raw_metrics(code)
	df.loc[i,'loc'] = raw.loc
	df.loc[i,'lloc'] = raw.lloc
	df.loc[i,'sloc'] = raw.sloc
	df.loc[i,'comments'] = raw.comments
	df.loc[i,'multi'] = raw.multi
	df.loc[i,'blank'] = raw.blank
	df.loc[i,'single_comments'] = raw.single_comments
    
	df.loc[i,'num_fns'] = number_of_functions(code)
	df.loc[i,'maintainability'] = maintainability_index(code)
    
    # ACCESS
	acc = access_calculator(code)
	df.loc[i,'private_access'] = acc['private']
	df.loc[i,'protected_access'] = acc['protected']
	df.loc[i,'public'] = acc['public']
    
    # FILE NAME
	df.loc[i,'file_name'] = file_name
    
	i+=1

df = df.fillna(0)
df = df.sort_values(by=['file_name'])
df.to_csv('semi_final.csv',index=False)

df2 = pd.read_csv('authors_list.csv')
df2 = df2.sort_values(by=['file_name'])

df_fin = pd.merge(df,df2,on='file_name')
df_fin = df_fin.fillna(0)
df_fin.to_csv('final_dataset.csv',index=False)
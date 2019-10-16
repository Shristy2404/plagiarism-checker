import code_metrics
import os 

input_dir = '/home/teddy/Desktop/IR/submissions/Lasso/'
data=[]

max_llc=0
max_lwc=0
columns = ['number_of_functions', 'maintainability_index', ]
for code in os.listdir(input_dir):
	code=input_dir+code
	if(len(code_metrics.line_length_calculator(code))>max_llc):
		max_llc=len(line_length_calculator(code))
	if(len(code_metrics.line_word_calculator(code))>max_lwc):
		max_lwc=len(line_word_calculator(code))

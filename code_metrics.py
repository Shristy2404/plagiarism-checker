from radon import raw 
from radon.visitors import ComplexityVisitor, HalsteadVisitor
from radon.complexity import cc_rank, cc_visit
from radon.metrics import h_visit , mi_visit
import re
import keyword
import radon

def halstead_metrics(input_file):
	with open(input_file, 'r') as file:
		return h_visit(file.read()).total

def raw_metrics(input_file):
	with open(input_file, 'r') as file:
		return raw.analyze(file.read())

def number_of_functions(input_file):
	with open(input_file, 'r') as file:
		v = ComplexityVisitor.from_code(file.read())
	return len(v.functions)

def maintainability_index(input_file):	
	with open(input_file,'r') as file:
		return radon.metrics.mi_visit(file.read(), multi=True)

def extract_code(input_file):
	with open(input_file, 'r') as file:
		l = (file.read()).splitlines()
			# l is the list of lines without leading or trailing whitespaces, tabs and new lines 
		for i in range(len(l)):
			l[i] = l[i].strip()
	return l 


###### Line Length Calculator 

def line_length_calculator(input_file):
	l=extract_code(input_file)
	max=0
	for i in range(len(l)):
		if(len(l[i])>max):
			max=len(l[i])
	llc=dict.fromkeys(range(max+1),0)
	for i in range(len(l)):
		llc[len(l[i])]+=1
	return llc

###### Line Word Calculator

def line_word_calculator(input_file):
	l=extract_code(input_file)
	max=0
	for i in range(len(l)):
		if(len(l[i].split())>max):
			max=len(l[i])
	lwc=dict.fromkeys(range(max+1),0)
	for i in range(len(l)):
		lwc[len(l[i].split())]+=1
	return lwc

##### Access Specifier Calculator 

def access_calculator(input_file):
	l=extract_code(input_file)
	acc=dict.fromkeys(["private", "protected", "public"],0)
	for i in range(len(l)):
		if("private" in l[i].split()):
			acc["private"]+=1
		if("public" in l[i].split()):
			acc["public"]+=1
		if("protected" in l[i].split()):
			acc["protected"]+=1
	if((acc["private"]+acc["public"]+acc["protected"])!=0):
		acc["private"]=(acc["private"]/(acc["private"]+acc["public"]+acc["protected"]))
		acc["public"]=(acc["public"]/(acc["private"]+acc["public"]+acc["protected"]))
		acc["protected"]=(acc["protected"]/(acc["private"]+acc["public"]+acc["protected"]))
	return acc

##### Identifiers Length Calculator 

# identifier = re.compile("^[^\d\W]\w*\Z")
# for i in range(len(l)):
# 	ind=l[i].find("=")
# 	if(ind>0):
# 		identifier=l[i][0:ind]
# 		print(identifier)
	# line=l[i].split()
	# if(not(line[0]=="#" and line[0]=="print")):
	# 	for word in line:
	# 		if(not(keyword.iskeyword(word)) and word.isidentifier()):
				# print(word)

##### Function names set 

# function_names = set()
# function_arguments = []
# for i in range(len(l)):
# 	line=l[i].split()
# 	if(len(line) > 0 and line[0]=='def'):
# 		ind = line[1].find('(')
# 		function_names.add(line[1][0:ind])
# 		arguments=[]
# 		ind1 = line[1].find(',')
# 		if(ind1!=-1):
# 			while(ind1!=-1):
# 				arguments.append(line[1][ind+1:ind1])
# 				ind1=line[1].find(',',ind1+1)
# 		ind2 = line[1].find(')')
# 		arguments.append(line[1][ind+1:ind2])
# 		function_arguments.append(arguments)
# print(function_names)
# print(function_arguments)


def short_updation(statement):
	if(statement.find("+")!=-1 or statement.find("-")!=-1 or statement.find("/")!=-1 or
	statement.find("*")!=-1 or statement.find("!")!=-1 or statement.find("%")!=-1 ):
		return True
	return False

def extract_identifier_names(input_file):
	id_list=set()
	with open(input_file,'r') as file:
		for line in file:
			split_line = re.split("=",line.strip())
			# print(split_line)
			try:
				_ = split_line[1]
				ids = split_line[0]
				if(not(ids.startswith("if") or ids.startswith("print")) and (ids.find(".")==-1)):
					if(not(short_updation(ids))):
						ind=ids.find('[')
						if(ind!=-1):
							id_list.add(ids[0:ind])
						else:
							for id in ids.strip().split(','):
								id_list.add(id.strip())	
			except IndexError: 
				continue
	return id_list

# print(halstead_metrics('knn.py'))
# print(raw_metrics('knn.py'))
# print(number_of_functions('knn.py'))
# print(maintainability_index('knn.py'))
# print(line_length_calculator('knn.py'))
# print(line_word_calculator('knn.py'))
# print(access_calculator('knn.py'))
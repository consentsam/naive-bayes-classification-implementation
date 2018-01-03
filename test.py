import time
import numpy
import pickle
start_time=time.time()
# Opening a File providing only the read access
f = open("r8-test-all-terms.txt","r")
# Reading all the lines in the file in a list
list_of_all_lines=f.readlines()
m=len(list_of_all_lines)


with open('vocablury.txt','r') as file_handler:
	vocablury=pickle.load(file_handler)

with open('y_labels.txt','r') as file_handler:
	y_labels=pickle.load(file_handler)

Phi_List=numpy.load('Phi_List.npy')

Theta_Matrix=numpy.load('Theta_Matrix.npy')

# Converting the data into appropriate input and keeping 
# number of count every word has in the article 
# and then extracting the values of the counter dictionary intov
# another list and then putting this list in another list which
# will ultimately form a list of list which will be the resultant matrix
Matrix_X=[]
Matrix_Y=[]
for i in list_of_all_lines:
	tmp=i.lower().split()
	x_inputs=tmp[1:]

	# Making a dictionary in which every element is mapped to 0
	dictionary_for_every_line=dict.fromkeys(vocablury,0)

	for j in x_inputs:
		if(j in vocablury):
			dictionary_for_every_line[j]=dictionary_for_every_line[j]+1
	# Extracting the list of values in the counter dictionary
	row_x=dictionary_for_every_line.values()
	Matrix_X.append(row_x)
	# Making a similar dictionary for y_labels in which the label which 
	# is tagged will be given a value 1

	y_labelled=tmp[0]
	dictionary_for_label_of_a_line=dict.fromkeys(y_labels,0)
	dictionary_for_label_of_a_line[y_labelled]=1
	row_y=dictionary_for_label_of_a_line.values()
	Matrix_Y.append(row_y)

Matrix_X=numpy.array(Matrix_X)
Matrix_Y=numpy.array(Matrix_Y)

Matrix_Y_into_Column_Y_can_be_called_as_original=[]
# Putting observed y_labels into a column 
for i in Matrix_Y:
	count=0
	for j in i:
		count=count+1
		if(j==1):
			Matrix_Y_into_Column_Y_can_be_called_as_original.append(count)
			break

with open('Matrix_Y_into_Column_Y_can_be_called_as_original.txt','w') as file_handler:
	pickle.dump(Matrix_Y_into_Column_Y_can_be_called_as_original,file_handler)

# Hessian Matrix
H_Matrix = numpy.log(Phi_List)+(Matrix_X.dot(numpy.log(Theta_Matrix)))

# The Given Data
Y_labels_extracted_from_test_data=[]
# Here we have classes 1,2,3,4,5,6,7,8 and so here count variable 
# will also be the class predicted 
for i in H_Matrix:
	max_element=numpy.amax(i)
	count=0
	for j in i:
		count=count+1
		if(j==max_element):
			Y_labels_extracted_from_test_data.append(count)
			break

with open('Y_labels_extracted_from_test_data.txt','w') as file_handler:
	pickle.dump(Y_labels_extracted_from_test_data,file_handler)

end_time = time.time()
print("Total Time Elapsed " +str(end_time-start_time))

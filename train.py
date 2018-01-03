import time
import numpy
import pickle
start_time=time.time()
#Opening a File providing only the read access
f = open("r8-train-all-terms.txt","rw+")
#Reading all the lines in the file in a list
list_of_all_lines=f.readlines()
m=len(list_of_all_lines)
#Initialising a vocablury in a set
vocablury = set()
y_labels = set()

for i in list_of_all_lines:
	#Extracting every word possible in the file
	#lower() converts every word in small so that 
	#'This' and 'this' are treated as the same word
	tmp = i.lower().split()
	y_labels.add(tmp[0])
	vocablury.update(tmp[1:])
#Converting into list so that pickle can write it as list easily
vocablury=list(vocablury)
y_labels=list(y_labels)


#Converting the data into appropriate input and keeping 
#number of count every word has in the article 
#and then extracting the values of the counter dictionary intov
#another list and then putting this list in another list which
#will ultimately form a list of list which will be the resultant matrix
Matrix_X=[]
Matrix_Y=[]
for i in list_of_all_lines:
	tmp=i.lower().split()
	x_inputs=tmp[1:]

	# Making a dictionary for every line  in which every  
	# element of dictionary is mapped to 0
	dictionary_for_every_line=dict.fromkeys(vocablury,0)

	for j in x_inputs:
		# Counting the Ocuurance of the word j in the line  
		dictionary_for_every_line[j]=dictionary_for_every_line[j]+1
	#Extracting the list of values in the counter dictionary
	row_x=dictionary_for_every_line.values()
	Matrix_X.append(row_x)

	#Making a similar dictionary for y_labels in which the label which 
	#is tagged will be given a value 1
	y_labelled=tmp[0]
	dictionary_for_label_of_a_line=dict.fromkeys(y_labels,0)
	dictionary_for_label_of_a_line[y_labelled]=1
	row_y=dictionary_for_label_of_a_line.values()
	Matrix_Y.append(row_y)

#Sum of column of the Matrix_Y
Matrix_Y=numpy.array(Matrix_Y)
y_label_column_sum=numpy.sum(Matrix_Y,axis=0)
y_label_column_sum=y_label_column_sum.astype(float)

#Calculating Phi(i)
Phi_List=numpy.divide(y_label_column_sum,m)

#Calculating Theta_Matrix
Matrix_X=numpy.array(Matrix_X).astype(float)
Theta_Matrix=Matrix_X.transpose().dot(Matrix_Y)

#Adding c=1 to every element 
# Laplace Smoothing 
Theta_Matrix=numpy.add(Theta_Matrix,1)

#Sum of elements in every column of X_transpose_Y
X_transpose_Y_column_sum=numpy.sum(Theta_Matrix,axis=0)

Theta_Matrix=numpy.divide(Theta_Matrix,X_transpose_Y_column_sum)
#.npy format is platform independent and in binary format 
#not understandable by us and more efficient

numpy.save('Theta_Matrix.npy',Theta_Matrix)

numpy.save('Phi_List.npy',Phi_List)

#Vocablury file save in a temporary file
with open('vocablury.txt','w') as file_handler:
	pickle.dump(vocablury,file_handler)
#y_labels file save in a temporary file
with open('y_labels.txt','w') as file_handler:
	pickle.dump(y_labels,file_handler)

end_time=time.time()
print("Time Elapsed = " + str(end_time-start_time))


# We have to save the vocablury set in a file because we are not allowed to
# update the vocablury according to the test data and any new word if 
# came then we ignore that word as if never existed


#-----------------------------------------------------------#
import pickle
import time
start_time=time.time()
# Observed Data/Classification
with open('Matrix_Y_into_Column_Y_can_be_called_as_original.txt','r') as file_handler:
	Matrix_Y_into_Column_Y_can_be_called_as_original=pickle.load(file_handler)
# Predicted Data/Classification
with open('Y_labels_extracted_from_test_data.txt','r') as file_handler:
	Y_labels_extracted_from_test_data=pickle.load(file_handler)

m1=len(Matrix_Y_into_Column_Y_can_be_called_as_original)
m2=len(Y_labels_extracted_from_test_data)
# Since m1 and m2 are same we can run loop on any one of them
count=0.0
for i in range(m1):
	if(Matrix_Y_into_Column_Y_can_be_called_as_original[i]==Y_labels_extracted_from_test_data[i]):
		count=count+1
accuracy=count/m1*100
print("The Accuracy of the above Model is "+str(accuracy))
end_time=time.time()
print("Time Elapsed "+str(end_time-start_time))
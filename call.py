from convolutional_neural_network import predection
j=1
while(j==1):
    n=input("Enter Image Name")
    d=input("Enter path")
    k=predection(d,n)
    print('\n',k)
    j=int(input("Enter 1 to continue"))

from os import listdir
from os.path import isfile, join
from os import walk

filenames = next(walk('./CV_results/s_logi'), (None, None, [])) [2]
#print(filenames)



iter = 1
k = 0
string_ = ''
sum_of_best = 0

for file_name in filenames:
    with open('./CV_results/s_logi/'+file_name) as the_file:
        lines = the_file.readlines()
        for i in range(len(lines)):
            if(lines[i][0] == '>'):
                if(iter > 9):
                    k = 1
                string_ = lines[i][28+k]+lines[i][29+k]+lines[i][30+k]+lines[i][31+k]+lines[i][32+k]+ lines[i][33+k]
                string_= string_.replace(',', '')
                string_= string_.replace(' ', '')
                string_= string_.replace('c', '')
                string_= string_.replace('f', '')
                #print(string_)
                number = float(string_)
                sum_of_best = sum_of_best + number
                iter = iter + 1

    #print(iter)
    #print('srednia:')
    #print(str(sum_of_best/(iter-1)))
    srednia = sum_of_best/(iter-1)
    with open('val_acc_result.txt', 'a') as yes:
        yes.write(file_name+': iter_val = '+str(iter)+', best avg = '+str(srednia)+'\n')

    iter = 0
    srednia = 0
    sum_of_best = 0
    string_ = ''


import pandas as pd
from matplotlib import pyplot as plt
import random
import numpy as np

# Make sure to use double backslashes or a raw string for the file path
file_path = ('D:\Book11.xlsx')

# Use 'xlsx' instead of 'slsx' for Excel files
dataset = pd.read_excel(file_path, header=None, names=['Date','Kasus harian','Total kasus'])

# Print the first few rows of the dataset
print(dataset.head())

# Check the data types of the columns
print(dataset.dtypes)

dataset['Kasus harian'] = pd.to_numeric(dataset['Kasus harian'], errors='coerce')
dataset['Total kasus'] = pd.to_numeric(dataset['Total kasus'], errors='coerce')

# Plotting
dataset.plot(x="Date", y="Kasus harian", title="grafik kasus harian")
#plt.show()

dataset.plot(x="Date", y="Total kasus", title="grafik Total kasus")
#plt.show()

#objective function

def ARIMA (miu, theta, dateindex):

  total_kasus = dataset['Total kasus'].to_list()  # Convert to list for easy indexing

  return miu + total_kasus[dateindex - 1] + theta * (total_kasus[dateindex - 1] - total_kasus[dateindex - 2])
  
#  return miu + dataset['Total kasus'][dateindex - 1] + theta*(dataset['Total kasus'][dateindex-1] - dataset['Total kasus'][dateindex-2])
  
#fitness Function

def fitfunc (predicted):
  totalAE=0
  for x in range(len(predicted)):
    totalAE += abs(dataset['Total kasus'][x]-predicted[x])
  MAE = totalAE/ len(predicted)
  return 1/MAE

#popsize
popsize = 50


population = []
for x in range(popsize):
  population.append([random.randint(10000,30000), round(random.uniform(1,5),2)])
population


predicted = [0]*len(population)
for index in range(len(population)):
  predicted[index]=[0]*50
  predicted[index][0]=dataset['Total kasus'][0]
  predicted[index][1]=dataset['Total kasus'][1]
  for x in range(2,50):
    predicted[index][x] = ARIMA(population[index][0], population[index][1],x)
for y in range(len(predicted)):
  predpd=pd.DataFrame(predicted[y])
  predpd.plot(title=("Prediksi" + str(y+1)))


fitness=[0]*len(population)
for index in range(len(population)):
  fitness[index] = round(fitfunc(predicted[index]),8)
fitness


datapop= population
datafit = fitness

print("Population:   ","Fitness Value:   " )
res = "\n".join("{} {}".format(x,y) for x,y in zip(datapop, datafit))
print(res)

pc = int(0.8*len(population))
pc



rou_list = list(zip(datapop,datafit))
random.shuffle(rou_list)
rpc = random.sample(rou_list, k=pc)

print("selected individu:   ")
ress = "\n".join("{}".format(x) for x in zip(rpc))
print(ress)




def crossover (population) :
  selected_pops = population
  n_offspring = int(round(len(selected_pops)/2,0)) 
  n_selected_pops = len(selected_pops)
  offspring = []
  avaliable_individu = [*range(0,n_selected_pops, 1)]
  alpha = 0.8

  cross_point = random.randint(1,2)


  print('Population before Crossover :')
  for x in range(n_selected_pops):
    print('Individu -','', x+1,'=',selected_pops[x][0])
  for i in range(n_offspring):
    get_individu_1 = random.choice(avaliable_individu) 
    while True:
      get_individu_2 = random.choice(avaliable_individu)
      if (get_individu_1 != get_individu_2):
        avaliable_individu.remove(get_individu_1) 
        avaliable_individu.remove(get_individu_2) 
        break

    random_parent_1 = selected_pops [get_individu_1]
    random_parent_2 = selected_pops [get_individu_2]
    miu_parent_1 = random_parent_1[0] [0]
    theta_parent_1 = random_parent_1[0] [1] 
    miu_parent_2 = random_parent_2[0] [0]
    theta_parent_2 = random_parent_2[0][1]

    if cross_point == 1 :
      new_miu_parent_1 = (miu_parent_1 *(1-alpha)) + (miu_parent_2 * alpha)
      new_miu_parent_2 = (miu_parent_1 * alpha) + (miu_parent_2 *(1-alpha)) 
      offspring.append([new_miu_parent_1, theta_parent_1])
      offspring.append([new_miu_parent_2,theta_parent_2])
    else:
      new_theta_parent_1 = (theta_parent_1 * (1-alpha)) + (theta_parent_2 * alpha)
      new_theta_parent_2 = (theta_parent_1 * alpha) + (theta_parent_2 *(1-alpha))
      offspring.append([miu_parent_1,new_theta_parent_1])
      offspring.append([miu_parent_2,new_theta_parent_2])

  return offspring

offspring_result = crossover(rpc)
print('population after crossover')
for x in range(len(offspring_result)):
    print('Individu -','', x+1,'=',offspring_result[x][0])


mutation_1=5
mutated = offspring_result
offspring_result[2][0]
mutated[2][0] = random.randint(10000,30000)
mutated


def mutation(offspring_result):
  mutation_result = offspring_result
  gnome_size = len(offspring_result)*len(offspring_result[0])
  Pm = 0.1
  number_mutation = int(round(Pm*gnome_size,0))

  for x in range(number_mutation):
    mutation_position = random.randint(1,gnome_size)
    individu_index = int(round(mutation_position/len(offspring_result[0]),0)) 
    left_over = mutation_position% len(offspring_result[0])
    if left_over != 0:
      new_miu = random.randint(10000,30000)
      mutation_result[individu_index - 1][0] = new_miu
    else:
      new_theta = round(random.uniform(1, 5), 2)
      mutation_result[individu_index - 1][1] = new_theta 
  return mutation_result

print(mutation(offspring_result))
print(mutation(offspring_result))




generation_num = 50
last_gen = []
for x in range(generation_num):
#calculating C(t) using current population 
    prediction=[0]*len(population) 
    for index in range(len(population)): 
      prediction[index]=[0]*50 
      prediction[index][0] = dataset['Total kasus'][0] 
      prediction[index][1] = dataset['Total kasus'] [1]
      for x in range(2, 50):
        prediction[index] [x] = ARIMA (population[index][0], population[index][1], x)

# current population fitness value
fitness [0]*len(population)
for index in range(len(population)): 
  fitness[index] = round(fitfunc(prediction[index]),8)
#roulette wheel selection 
  rou_list=list(zip(population, fitness)) 
  random.shuffle(rou_list) 
  rpc=random.choices(rou_list, k=pc)
#offspring crossover 
offspring=crossover(rpc)

#offspring mutation
offspring_mutated = mutation(offspring)

#calculating C(t) using mutated offspring
prediction_offspring=[0]*len(population)
for index in range(len(population)):
  prediction_offspring [index]=[0]*50
  prediction_offspring [index][0] = dataset['Total kasus'][0] 
  prediction_offspring [index][1] = dataset['Total kasus'][1]
  for x in range(2, 50):
    prediction_offspring [index][x] = ARIMA(population[index][0], population[index][1], x)


# offspring fitness value
offspring_fitness = [0] * len(offspring)
for index in range(len(offspring)):
    #predicted = ARIMA(offspring[index][0], offspring[index][1], offspring [index]) 
   # offspring_fitness[index] = round(fitfunc(predicted), 8)
    # offspring_fitness[index] = round(fitfunc(predicted.to_list()), 8)


# #offspring fitness value
# offspring_fitness = [0]* len(offspring)
# for index in range(len(offspring)):
   offspring_fitness [index] = round(fitfunc(prediction_offspring[index]),8)

offspring_list = list(zip(offspring_mutated, offspring_fitness))

#get best 15 individo from parent offspring
all_individu= rpc + offspring_list 
all_individu_sorted = sorted(all_individu, key=lambda x: x[1], reverse=True)

new_population = all_individu_sorted[0:popsize]

#update population

population = [i[0] for i in new_population]
last_gen = new_population

print('Individu\tFitness Value')
last_gen

# day 51 and 52 prediction
final_prediction = [0] *52
final_prediction[0] = dataset['Total kasus'][0]
final_prediction[1] = dataset['Total kasus'][1]
for x in range(2, 52):
  final_prediction[x] = ARIMA(population[0][0], population[0][1], x)
final_prediction

finalpredpd = pd.DataFrame(final_prediction)
finalpredpd.plot(title=("Final Prediction")) 
dataset.plot(x="Date", y="Total kasus", title="Total kasus chart")

print('Prediction Day 51:')
print(int(round(final_prediction[50]-10000, 0)))
print('Prediction Day 52:')
print(int(round(final_prediction[51]-10000, 0)))
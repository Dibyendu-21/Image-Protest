# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:19:36 2019

@author: Sonu
"""
import matplotlib.font_manager as font_manager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
df_original=pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\trainer.csv',nrows=1785)
vio=df_original.loc[df_original['protest']==1,'violence']
y=np.array(vio)
violnce_score=np.empty(len(y), dtype =int)
z=np.empty(len(y), dtype =int)
violnce_score1=np.empty(len(y), dtype =int)
for k in range(0,len(vio)):
        y[k]=float(y[k]) 
        z[k]=k
print(len(z))
print(len(y))
mean_vio=np.mean(y)

print(mean_vio)

#new_viol_data=pd.dataframe(y, columns=['orig_score'])

yy=vio.rolling(window=100).mean()

plt.rcParams['axes.labelweight'] = 'bold'
plt.plot(z, yy , color='red')
plt.xlabel('Number of samples',size = 13)
plt.ylabel('Violence Score',size = 14)
plt.title('Original Violence Score',size = 25)
plt.xticks(size = 10, weight='bold')
plt.yticks(size = 10, weight='bold')
plt.show()


df_sign =pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Important programs\Result\Predicted_score\violence_test_sign_complete.csv')
df_photo =pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Important programs\Result\Predicted_score\violence_test_photo_complete.csv')
df_fire =pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Important programs\Result\Predicted_score\violence_test_fire_complete.csv')
df_police =pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Important programs\Result\Predicted_score\violence_test_police_complete.csv')
df_children = pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Important programs\Result\Predicted_score\violence_test_children_complete.csv')
df_group_20 =pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Important programs\Result\Predicted_score\violence_test_group_20_complete.csv')
df_group_100 =pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Important programs\Result\Predicted_score\violence_test_group_100_complete.csv')
df_flag =pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Important programs\Result\Predicted_score\violence_test_flag_complete.csv')
df_night =pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Important programs\Result\Predicted_score\violence_test_night_complete.csv')
df_shouting =pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Important programs\Result\Predicted_score\violence_test_shouting_complete.csv')

sign_score=df_sign['sign_score']
photo_score=df_photo['photo_score']
fire_score=df_fire['fire_score']
police_score=df_police['police_score']
children_score=df_children['children_score']
group_20_score=df_group_20['group_20_score'] 
group_100_score=df_group_100['group_100_score'] 
flag_score=df_flag['flag_score']
night_score=df_night['night_score']
shouting_score=df_shouting['shouting_score']

#weight=np.empty(len(10),dtype=object)
violence_score=np.empty(len(y),dtype=object)
print(len(violence_score))
#weight[0]= 0.16413992
#weight[1]= -0.00391488
#weight[2]= 0.4592643
#weight[3]=  0.28174789
#weight[4]= 0.01663969
#weight[5]= 0.17136479
#weight[6]= 0.04395227
#weight[7]= 0.07411503
#weight[8]= 0.09506399
#weight[9]= 0.09376591





for i in range(0,len(violence_score)):
        sign_score[i]=(sign_score[i]*0.16413992)
        photo_score[i]=(photo_score[i]*-0.00391488)
        fire_score[i]=(fire_score[i]*0.4592643)
        police_score[i]=(police_score[i]*0.28174789)
        children_score[i]=(children_score[i]*0.01663969)
        group_20_score[i]=(group_20_score[i]*0.17136479)
        group_100_score[i]=(group_100_score[i]*0.04395227)
        flag_score[i]=(flag_score[i]*0.07411503)
        night_score[i]=(night_score[i]*0.09506399)
        shouting_score[i]=(shouting_score[i]*0.09376591)
        violnce_score1[i]=sign_score[i]+photo_score[i]+ fire_score[i]+police_score[i]+children_score[i]+group_20_score[i]+group_100_score[i]+flag_score[i]+night_score[i]+shouting_score[i]




violence_score1=pd.DataFrame(violnce_score, columns=['violence_score'])
if not os.path.isdir('Result'):
    os.mkdir('Result')
sub_file1 = os.path.join('Result', 'violence_score' + '.csv')
violence_score1.to_csv(sub_file1, index=False)

violence_score1 = pd.read_csv(r'C:\Users\Sonu\Documents\M.TECH\Research\UCLA-protest\img\new\Important_programs\Result\violence_score.csv')


vio_score=violence_score1['violence_score']
mean_new=np.mean(vio_score)
print(mean_new)
yy_new=vio_score.rolling(window=100).mean()


plt.plot(z, yy_new , color='green')
plt.rcParams['axes.labelweight'] = 'bold'
plt.xlabel('Number of samples',size = 13)
plt.ylabel('Violence Score',size = 14)
plt.title('Calculated Violence Score',size = 25)
plt.xticks(size = 10, weight='bold')
plt.yticks(size = 10, weight='bold')
plt.show()

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
plt.suptitle('Violence comparision between actual and calculated violence')
plt.plot(z, yy , color='red')
plt.ylabel('Violence Score')
ax[1].plot(z, yy_new , color='green') 
plt.xlabel('Number of samples')
#fig.text(0.5, 0.04, 'Number of samples', ha='center')
#fig.text(0.04, 0.5, 'Violence Score', va='center', rotation='vertical')
plt.show()


#fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
#plt.title('Violence comparision between actual and calculated violence',size = 12, weight='bold')
plt.rcParams['axes.labelweight'] = 'bold'
#plt.legend(prop={'weight':'bold'})
font = font_manager.FontProperties(family='Comic Sans MS',weight='bold',style='normal', size=6.124)
plt.plot(z, yy , color='red', label='Original Violence Score')
plt.ylabel('Violence Score',size = 14)
plt.plot(z, yy_new , color='green', label='Calculated Violence Score') 
plt.xlabel('Number of samples',size = 13)
plt.legend(prop=font,bbox_to_anchor=(0.01, 1.01),loc='upper left')
plt.xticks(size = 10, weight='bold')
plt.yticks(size = 10, weight='bold')
#fig.text(0.5, 0.04, 'Number of samples', ha='center')
#fig.text(0.04, 0.5, 'Violence Score', va='center', rotation='vertical')
plt.show()